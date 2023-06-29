import os
from pathlib import Path
import shutil
import zipfile
import tarfile
import sys
import logging
import traceback
import statistics
import numpy as np
import pydicom as pydcm
import json
import SimpleITK as sITK
import skimage
import score_autocontours_lib
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


def rename_structures(structure_data, dictionary_filename):
    # Renames the structures according to the dictionary
    with open(dictionary_filename) as f:
        structure_dictionary = json.load(f)

    for idx, ref_roi in enumerate(structure_data.StructureSetROISequence):
        ref_name = ref_roi.ROIName
        output_name = ref_name
        for dict_key, dict_variants in structure_dictionary.items():
            if ref_name.lower().strip() == dict_key.lower().strip():
                output_name = dict_key
                break
            else:
                for variant in dict_variants:
                    if ref_name.lower().strip() == variant.lower().strip():
                        output_name = dict_key
                        break
        structure_data.StructureSetROISequence[idx].ROIName = output_name

    return structure_data


def errprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_case_id(cases, frame_of_reference, file_type, file_checked):
    case_id = -1
    for case in cases:
        if cases[case]['image_for'] == frame_of_reference:
            case_id = cases[case]['number']
    if case_id == -1:
        logger.warning(f"Could not find matching Frame of Reference for {file_type} file: {file_checked}")
        # errprint(f"Could not find matching Frame of Reference for {file_type} ffile: {file_checked}")
    return case_id


def subtract_structures(structure_masks, primary_struct_name, struct_to_subtract, new_struct_name,
                        image_volume):
    if (primary_struct_name in structure_masks) & (struct_to_subtract in structure_masks):
        primary_struct = structure_masks[primary_struct_name]
        secondary_struct = structure_masks[struct_to_subtract]
        primary_mask = sITK.GetArrayFromImage(primary_struct)
        secondary_mask = sITK.GetArrayFromImage(secondary_struct)
        print("Creating: {},".format(new_struct_name, end=''))
        # print(' primary count=', len(primary_mask[primary_mask[:, :, :] == 1]))
        # print(' secondary count=', len(secondary_mask[secondary_mask[:, :, :] == 1]))
        new_mask = primary_mask - secondary_mask
        print(' number of voxels: {}'.format(len(new_mask[new_mask[:, :, :] == 1])))
        new_mask[new_mask < 0] = 0

        segmentation = sITK.GetImageFromArray(new_mask)
        segmentation.SetOrigin(image_volume.GetOrigin())
        segmentation.SetSpacing(image_volume.GetSpacing())
        segmentation.SetDirection(image_volume.GetDirection())
        segmentation.SetMetaData("Structure Name", new_struct_name)
        return segmentation
    else:
        if primary_struct_name in structure_masks:
            # return primary as there was nothing to subtract
            primary_struct = structure_masks[primary_struct_name]
            primary_mask = sITK.GetArrayFromImage(primary_struct)

            segmentation = sITK.GetImageFromArray(primary_mask)
            segmentation.SetOrigin(image_volume.GetOrigin())
            segmentation.SetSpacing(image_volume.GetSpacing())
            segmentation.SetDirection(image_volume.GetDirection())
            segmentation.SetMetaData("Structure Name", new_struct_name)
            return segmentation
        else:
            # need to return an empty object?
            vol_size = image_volume.GetSize()
            voxel_mask_data = np.zeros([vol_size[0], vol_size[1], vol_size[2]], dtype=np.uint8)

            segmentation = sITK.GetImageFromArray(np.moveaxis(voxel_mask_data, [0, 1, 2], [2, 1, 0]))
            segmentation.SetOrigin(image_volume.GetOrigin())
            segmentation.SetSpacing(image_volume.GetSpacing())
            segmentation.SetDirection(image_volume.GetDirection())
            segmentation.SetMetaData("Structure Name", new_struct_name)
            return segmentation


def get_dvhs(dose_volume, structure_masks):
    # organ_dvhs = self.get_dvhs(resampled_dose, gt_structs)
    organ_dvhs = {}
    for struct_name in structure_masks:
        mask = structure_masks[struct_name]
        dose_as_array = sITK.GetArrayFromImage(dose_volume)
        mask_as_array = sITK.GetArrayFromImage(mask)
        if np.sum(mask_as_array)>0:
            dose_values = dose_as_array[mask_as_array == 1]
            dose_values.sort()
            organ_dvhs[struct_name] = dose_values
    return organ_dvhs


class Autortp:
    def __init__(self):
        if os.name == 'nt':
            # Running on my machine locally
            local_path = os.path.dirname(os.path.abspath(__file__))
            self.ground_truth_path = os.path.join(local_path, "ground-truth")
            self.predictions_path = os.path.join(local_path, "..", "test")
        else:
            # Running on the grand challenges docker
            self.ground_truth_path = Path("./ground-truth")
            self.predictions_path = Path("../../input/")

        self.dictionary = "dictionary.json"
        self.output_path = "../../output"
        self.output_file = "../../output/metrics.json"
        self.number_of_cases = 3  # TODO This failed on me as I have to manually update it. Can I do that automatically?
        self.ground_truth_cases = {}
        self.test_cases = {}
        self.required_label_map = {'Bladder': 0, 'Rectum': 1, 'Prostate': 2, 'Femur_Head_R': 3, 'Femur_Head_L': 4,
                                   'Bowel_Bag': 5, 'PTVp_7400': 6, 'PTVp_7100': 7, 'PTVp_6000': 8, 'PTVp_6600': 8,
                                   'PTVn_6000': 9, 'PTVn_5000': 10, 'CTV_Prostate': 11, 'CTV_ProstateBed': 12,
                                   'CTV_SeminalVes': 13, 'CTV_LN_Pelvic': 14}

    def validate(self):
        # This function parses the data to check that we have all that we need
        # Check if we have any zip file and extract them

        for curr_dir, sub_dirs, files in os.walk(self.predictions_path):
            if len(files) > 0:
                for testfile in files:
                    full_testfile_name = os.path.join(curr_dir, testfile)
                    logger.warning(f"Input file: {full_testfile_name}")
                    if zipfile.is_zipfile(full_testfile_name) | tarfile.is_tarfile(full_testfile_name):
                        shutil.unpack_archive(full_testfile_name, self.predictions_path)
            else:
                errprint(f"No input files received.")
                return 1

        # Now we query the CTs from the GT directory, so gather the details
        # We are going to assume this data is structured correctly in folders as
        # this is the data we are creating. Will be one folder per case, containing
        # one folder for the image and one RTSS.dcm file.
        for i in range(1, self.number_of_cases + 1):
            case_name = "AUTO-RTP Case {:02d}".format(i)
            case = {
                'number': i,
                'name': case_name,  # "AUTO-RTP Case " + str(i),
                'rtstruct_id_0': "Consensus.dcm",
                'rtstruct_id_1': "Expert1.dcm",
                'rtstruct_id_2': "Expert2.dcm",
                'rtstruct_id_3': "Expert3.dcm"
            }
            case['folder'] = Path(self.ground_truth_path, case['name'])
            case['image_folder'] = Path(case['folder'], "CT")
            case['rtstruct_file_0'] = Path(case['folder'], "RTSS", case['rtstruct_id_0'])
            case['rtstruct_file_1'] = Path(case['folder'], "RTSS", case['rtstruct_id_1'])
            case['rtstruct_file_2'] = Path(case['folder'], "RTSS", case['rtstruct_id_2'])
            case['rtstruct_file_3'] = Path(case['folder'], "RTSS", case['rtstruct_id_3'])

            for image_file in os.listdir(case['image_folder']):
                image_path = Path(case['image_folder'], image_file)
                if os.path.isfile(image_path):
                    # this should be an image
                    with open(image_path, "rb") as f:
                        try:
                            ds = pydcm.dcmread(f)
                            if (ds['SOPClassUID'].value == "1.2.840.10008.5.1.4.1.1.2") | (
                                    ds['SOPClassUID'].value == "1.2.840.10008.5.1.4.1.1.4"):  # CT or MR
                                case['image_series_uid'] = str(ds['SeriesInstanceUID'].value)
                                case['image_for'] = str(ds['FrameOfReferenceUID'].value)
                                break  # only want to get the info once
                            else:
                                logger.warning(f"GT image not CT or MR: {image_file}")
                                # errprint(f"GT image not CT or MR:  {image_file}")
                        except InvalidDicomError:
                            logger.warning(f"Failed to read GT image {image_file}")
                            # errprint(f"Failed to read GT image {f}")
                else:
                    logger.warning(f"Failed to read GT image {image_file}")
                    # errprint(f"Failed to read GT image {image_file}")

            if case['image_for'] is None:
                logger.warning(f"Failed to read any GT image in {case['image_folder']}")
                errprint(f"Failed to read any GT image in {case['image_folder']}")
                return 1

            # Now parse the GT RTSS
            if os.path.isfile(case['rtstruct_file_0']):
                # this should be an RTSS dicom
                with open(case['rtstruct_file_0'], "rb") as f:
                    try:
                        ds = pydcm.dcmread(f)
                        if ds['SOPClassUID'].value == "1.2.840.10008.5.1.4.1.1.481.3":  # RTSS
                            case['rtstruct_referenced_series'] = str(
                                ds['ReferencedFrameOfReferenceSequence'].value[0]['RTReferencedStudySequence'].value[0][
                                    'RTReferencedSeriesSequence'].value[0]['SeriesInstanceUID'].value)
                            case['rtstruct_referenced_for'] = str(
                                ds['ReferencedFrameOfReferenceSequence'].value[0]['FrameOfReferenceUID'])
                        else:
                            logger.error(f"GT contour file is not DICOM RTSTRUCT: {case['rtstruct_file_0']}")
                            errprint(f"GT contour file is not DICOM RTSTRUCT:  {case['rtstruct_file_0']}")
                            return 1
                    except InvalidDicomError:
                        logger.error(f"Failed to read GT RTSTRUCT {case['rtstruct_file_0']}")
                        errprint(f"Failed to read GT RTSTRUCT {case['rtstruct_file_0']}")
                        return 1
            else:
                logger.error(f"Failed to read GT RTSTRUCT {case['rtstruct_file_0']}")
                errprint(f"Failed to read GT RTSTRUCT {case['rtstruct_file_0']}")
                return 1

            self.ground_truth_cases[i] = case

        for gt_case in self.ground_truth_cases:
            case = {
                'number': self.ground_truth_cases[gt_case]['number'],
                'name': self.ground_truth_cases[gt_case]['name']
            }
            self.test_cases[case['number']] = case

        # Now we parse the results directory looking for Dicom files
        for curr_dir, sub_dirs, files in os.walk(self.predictions_path):
            if len(files) > 0:
                for testfile in files:
                    file_path = Path(curr_dir, testfile)
                    if os.path.isfile(file_path):
                        # this should be a dicom object
                        with open(file_path, "rb") as f:
                            try:
                                ds = pydcm.dcmread(f)
                                if ds['SOPClassUID'].value == "1.2.840.10008.5.1.4.1.1.481.2":
                                    # RTDOSE SOP Class UID = 1.2.840.10008.5.1.4.1.1.481.2
                                    # Dose should have same FOR to match CT
                                    # Dose has Referenced RT plan SOP instance UID to link to plan
                                    # Dose should be in GY not RELATIVE
                                    dose_for = str(ds['FrameOfReferenceUID'].value)
                                    case_id = get_case_id(self.ground_truth_cases, dose_for, 'Dose', file_path)
                                    if case_id != -1:
                                        self.test_cases[case_id]['dose_file'] = file_path
                                        self.test_cases[case_id]['dose_for'] = dose_for
                                        self.test_cases[case_id]['dose_uid'] = str(
                                            ds['SOPInstanceUID'].value)
                                        self.test_cases[case_id]['dose_referenced_plan'] = \
                                            ds['ReferencedRTPlanSequence'].value[0]['ReferencedSOPInstanceUID'].value
                                        if ds['DoseUnits'] == "RELATIVE":
                                            logger.error(
                                                f"Please use GY value units for "
                                                f"{self.test_cases[case_id]['dose_file']}")
                                            errprint(
                                                f"Please use GY value units for "
                                                f"{self.test_cases[case_id]['dose_file']}")
                                            return 1
                                elif ds['SOPClassUID'].value == "1.2.840.10008.5.1.4.1.1.481.3":
                                    # RTSTRUCT SOP Class UID = 1.2.840.10008.5.1.4.1.1.481.3
                                    # Referenced FOR Sequence
                                    #   - FOR UID to match CT
                                    #   - RT Referenced Series Sequence
                                    #       - Series Instance UID to match CT
                                    rtstruct_for = str(
                                        ds['ReferencedFrameOfReferenceSequence'].value[0]['FrameOfReferenceUID'].value)
                                    case_id = get_case_id(self.ground_truth_cases, rtstruct_for, 'RTSTRUCT', file_path)
                                    if case_id != -1:
                                        self.test_cases[case_id]['rtstruct_file'] = file_path
                                        self.test_cases[case_id]['rtstruct_for'] = rtstruct_for
                                        self.test_cases[case_id]['rtstruct_uid'] = str(
                                            ds['SOPInstanceUID'].value)
                                        self.test_cases[case_id]['rtstruct_referenced_series'] = str(
                                            ds['ReferencedFrameOfReferenceSequence'].value[0][
                                                'RTReferencedStudySequence'].value[0][
                                                'RTReferencedSeriesSequence'].value[0]['SeriesInstanceUID'].value)
                                elif ds['SOPClassUID'].value == "1.2.840.10008.5.1.4.1.1.481.5":
                                    # PLAN SOP Class UID = 1.2.840.10008.5.1.4.1.1.481.5
                                    # Referenced Structure Set Sequence
                                    #   - Referenced SOP Instance UID to match RTSTRUCT SOP Instance UID
                                    # Frame of Reference UID to match CT
                                    plan_for = str(ds['FrameOfReferenceUID'].value)
                                    case_id = get_case_id(self.ground_truth_cases, plan_for, 'Plan', file_path)
                                    if case_id != -1:
                                        self.test_cases[case_id]['plan_file'] = file_path
                                        self.test_cases[case_id]['plan_for'] = plan_for
                                        self.test_cases[case_id]['plan_uid'] = str(
                                            ds['SOPInstanceUID'].value)
                                        self.test_cases[case_id]['plan_referenced_rtstruct'] = str(
                                            ds['ReferencedStructureSetSequence'].value[0][
                                                'ReferencedSOPInstanceUID'].value)
                                else:
                                    logger.warning(f"Unrecognised/needed dicom file: {file_path}")
                                    # errprint(f"Unrecognised/needed dicom file:  {file_path}")
                            except InvalidDicomError:
                                logger.warning(f"Unrecognised/unneeded non-dicom file: {file_path}")
                                # errprint(f"Unrecognised/needed non-dicom file:  {file_path}")
                    else:
                        logger.warning(f"Not a file: {file_path}")
                        # errprint(f"Not a file  {file_path}")

        # Now we have to check all is present and correct
        has_everything = True
        scores = {
            'cases': {},
            'total': 0
        }
        total_score = 0
        for case_number in self.test_cases:
            case = self.test_cases[case_number]
            case_success = True
            confirm_matches = True
            if 'rtstruct_for' not in case:
                # logger.error(f"No RTSTRUCT found for {case['name']}")
                errprint(f"No RTSTRUCT found for {case['name']}")
                has_everything = False
                case_success = False
                confirm_matches = False
            if 'plan_for' not in case:
                # logger.error(f"No RTPLAN found for {case['name']}")
                errprint(f"No RTPLAN found for {case['name']}")
                has_everything = False
                case_success = False
                confirm_matches = False
            if 'dose_for' not in case:
                # logger.error(f"No RTDOSE found for {case['name']}")
                errprint(f"No RTDOSE found for {case['name']}")
                has_everything = False
                case_success = False
                confirm_matches = False
            if confirm_matches:
                # the get_case_id has already checked we match FORs matched the CT, therefore they match each other
                if case['rtstruct_referenced_series'] != self.ground_truth_cases[case['number']]['image_series_uid']:
                    # logger.error(f"RTSTRUCT for {case['name']} does reference correct image series")
                    errprint(f"RTSTRUCT for {case['name']} does reference correct image series")
                    has_everything = False
                    case_success = False
                if case['plan_referenced_rtstruct'] != case['rtstruct_uid']:
                    # logger.error(f"RTPLAN for {case['name']} does reference correct RTSTRUCT series")
                    errprint(f"RTPLAN for {case['name']} does reference correct RTSTRUCT series")
                    has_everything = False
                    case_success = False
                if case['dose_referenced_plan'] != case['plan_uid']:
                    # logger.error(f"RTDOSE for {case['name']} does reference correct RTPLAN series")
                    errprint(f"RTDOSE for {case['name']} does reference correct RTPLAN series")
                    has_everything = False
                    case_success = False
            if case_success:
                scores['cases'][f'case{case_number}'] = 1
                total_score = total_score + 1
            else:
                scores['cases'][f'case{case_number}'] = 0

        scores['total'] = total_score
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        with open(self.output_file, "w") as outfile:
            json.dump(scores, outfile, indent=4)
            outfile.close()

        if has_everything:
            return 0
        else:
            return 1

    def evaluate(self):
        # This function will run through the cases and score them
        scores = {
            'cases': {},
            'total': 0,
            'own_contour': 0
        }
        total_score = 0
        own_contour = 0
        number_of_cases = 0
        number_of_participant_cases = 0
        # for case in self.ground_truth_cases.values():
        for case_number in self.ground_truth_cases:
            case = self.ground_truth_cases[case_number]
            scores['cases'][f'case {case_number}'] = {}
            for contour_set_number in range(-1, 4):  # -1 is the test set, 0 is the consensus, 1-3 are the experts
                if contour_set_number == 0:
                    scores['cases'][f'case {case_number}']['Consensus'] = \
                        self.score_case(case['number'], contour_set=contour_set_number)
                    total_score = total_score + scores['cases'][f'case {case_number}']['Consensus']['Overall']
                    number_of_cases = number_of_cases + 1
                elif contour_set_number == -1:
                    scores['cases'][f'case {case_number}']['Participant'] = \
                        self.score_case(case['number'], contour_set=contour_set_number)
                    own_contour = own_contour + scores['cases'][f'case {case_number}']['Participant']['Overall']
                    number_of_participant_cases = number_of_participant_cases + 1
                else:
                    scores['cases'][f'case {case_number}'][f'Expert {contour_set_number}'] = self.score_case(
                        case['number'],
                        contour_set=contour_set_number)

        scores['total'] = total_score / number_of_cases
        scores['own_contour'] = own_contour / number_of_participant_cases

        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        with open(self.output_file, "w") as outfile:
            json.dump(scores, outfile, indent=4)
            outfile.close()

        return 0

    def score_case(self, case_number, contour_set=0):

        # TODO: Nasty hardcoded debugging
        output_dvh_values = False

        # This method scores each individual case
        print("Scoring case: {}".format(case_number))
        case_scores = {}

        ct_volume = self.load_ct(case_number)
        dose_volume = self.load_dose(case_number)
        resampled_dose = sITK.Resample(dose_volume, ct_volume, sITK.Transform(), sITK.sitkLinear, 0)
        # test segmentations are not used for scoring dose
        # contour measures will be calculated from the RTSS
        # test_structs = self.load_rtstructs(case_number, ct_volume, self.required_label_map, test=True)
        if contour_set >= 0:
            gt_structs = self.load_rtstructs(case_number, ct_volume, self.required_label_map,
                                             test=False, expert=contour_set, rename=True)
        else:
            gt_structs = self.load_rtstructs(case_number, ct_volume, self.required_label_map,
                                             test=True, rename=True)

        # Renaming structures according to dictionary should be done by the loading function
        # gt_structs = rename_structures(gt_structs, self.dictionary)

        ct_resolution = ct_volume.GetSpacing()
        voxel_volume_in_cc = (ct_resolution[0] * ct_resolution[1] * ct_resolution[2]) / 1000

        # Do boolean operations on the PTVs so we measure the DVH against the right objects
        if case_number in [1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15]:
            gt_structs["PTVp_7100_dvh"] = subtract_structures(gt_structs, "PTVp_7100", "PTVp_7400", "PTVp_7100_dvh",
                                                              ct_volume)
            gt_structs["PTVp_6000_dvh"] = subtract_structures(gt_structs, "PTVp_6000", "PTVp_7100", "PTVp_6000_dvh",
                                                              ct_volume)

        organ_dvhs = get_dvhs(resampled_dose, gt_structs)
        # This is nasty coding. Ideally the constraints and objectives would all be read from a file
        # but for the sake of speed of implementation, this is just hard-coded for now

        number_of_oars = 0.0
        case_scores['OARs'] = 0.0
        number_of_targets = 0.0
        case_scores['Targets'] = 0.0

        # Bowel_Bag
        if 'Bowel_Bag' in organ_dvhs:
            number_of_oars = number_of_oars + 1
            dvh = organ_dvhs['Bowel_Bag']
            # V45 Optimal 78cc Mandatory 158cc
            # v45 = len(dvh[dvh >= 45]) / len(dvh)
            vol_in_cc_v45 = len(dvh[dvh >= 45]) * voxel_volume_in_cc
            case_scores['Bowel_Bag v45'] = min(max((158.0 - vol_in_cc_v45) / 80.0, 0.0), 1.0)
            # V50 Optimal 17cc Mandatory 110cc
            # v50 = len(dvh[dvh >= 50]) / len(dvh)
            vol_in_cc_v50 = len(dvh[dvh >= 50]) * voxel_volume_in_cc
            case_scores['Bowel_Bag v50'] = min(max((110.0 - vol_in_cc_v50) / 93.0, 0.0), 1.0)
            # V55 Optimal 14cc Mandatory 28cc
            # v55 = len(dvh[dvh >= 55]) / len(dvh)
            vol_in_cc_v55 = len(dvh[dvh >= 55]) * voxel_volume_in_cc
            case_scores['Bowel_Bag v55'] = min(max((28.0 - vol_in_cc_v55) / 14.0, 0.0), 1.0)
            # V60 Optimal 0.5cc Mandatory 6cc
            # v60 = len(dvh[dvh >= 60]) / len(dvh)
            vol_in_cc_v60 = len(dvh[dvh >= 60]) * voxel_volume_in_cc
            case_scores['Bowel_Bag v60'] = min(max((6.0 - vol_in_cc_v60) / 5.5, 0.0), 1.0)
            # V65 Optimal 0cc Mandatory 0.5cc
            # v65 = len(dvh[dvh >= 65]) / len(dvh)
            vol_in_cc_v65 = len(dvh[dvh >= 65]) * voxel_volume_in_cc
            case_scores['Bowel_Bag v65'] = min(max((0.5 - vol_in_cc_v65) / 0.5, 0.0), 1.0)
            case_scores['Bowel_Bag'] = (case_scores['Bowel_Bag v45'] + case_scores['Bowel_Bag v50'] +
                                        case_scores['Bowel_Bag v55'] + case_scores['Bowel_Bag v60'] +
                                        case_scores['Bowel_Bag v65'])
            case_scores['OARs'] = case_scores['OARs'] + case_scores['Bowel_Bag']

            if output_dvh_values:
                case_scores['Bowel_Bag v45'] = vol_in_cc_v45
                case_scores['Bowel_Bag v50'] = vol_in_cc_v50
                case_scores['Bowel_Bag v55'] = vol_in_cc_v55
                case_scores['Bowel_Bag v60'] = vol_in_cc_v60
                case_scores['Bowel_Bag v65'] = vol_in_cc_v65
        else:
            case_scores['Bowel_Bag v45'] = 'n/a'
            case_scores['Bowel_Bag v50'] = 'n/a'
            case_scores['Bowel_Bag v55'] = 'n/a'
            case_scores['Bowel_Bag v60'] = 'n/a'
            case_scores['Bowel_Bag v65'] = 'n/a'
            case_scores['Bowel_Bag'] = 'n/a'

        # Rectum
        if 'Rectum' in organ_dvhs:
            number_of_oars = number_of_oars + 1
            dvh = organ_dvhs['Rectum']
            # V50 Optimal 50% Mandatory 60%
            v50 = len(dvh[dvh >= 50]) / len(dvh)
            case_scores['Rectum v50'] = min(max((0.6 - v50) / 0.1, 0.0), 1.0)
            # V60 Optimal 35% Mandatory 50%
            v60 = len(dvh[dvh >= 60]) / len(dvh)
            case_scores['Rectum v60'] = min(max((0.5 - v60) / 0.15, 0.0), 1.0)
            # V65 Optimal 25% Mandatory 30%
            v65 = len(dvh[dvh >= 65]) / len(dvh)
            case_scores['Rectum v65'] = min(max((0.3 - v65) / 0.05, 0.0), 1.0)
            # V70 Optimal 10% Mandatory 15%
            v70 = len(dvh[dvh >= 70]) / len(dvh)
            case_scores['Rectum v70'] = min(max((0.15 - v70) / 0.05, 0.0), 1.0)
            # V75 Optimal 3% Mandatory 5%
            v75 = len(dvh[dvh >= 75]) / len(dvh)
            case_scores['Rectum v75'] = min(max((0.05 - v75) / 0.02, 0.0), 1.0)
            case_scores['Rectum'] = (case_scores['Rectum v50'] + case_scores['Rectum v60'] +
                                     case_scores['Rectum v65'] + case_scores['Rectum v70'] +
                                     case_scores['Rectum v75'])
            case_scores['OARs'] = case_scores['OARs'] + case_scores['Rectum']

            if output_dvh_values:
                case_scores['Rectum v50'] = v50
                case_scores['Rectum v60'] = v60
                case_scores['Rectum v65'] = v65
                case_scores['Rectum v70'] = v70
                case_scores['Rectum v75'] = v75
        else:
            case_scores['Rectum v50'] = 'n/a'
            case_scores['Rectum v60'] = 'n/a'
            case_scores['Rectum v65'] = 'n/a'
            case_scores['Rectum v70'] = 'n/a'
            case_scores['Rectum v75'] = 'n/a'
            case_scores['Rectum'] = 'n/a'

        # Bladder
        if 'Bladder' in organ_dvhs:
            number_of_oars = number_of_oars + 1
            dvh = organ_dvhs['Bladder']
            # V50 Optimal 50% Mandatory 100%
            v50 = len(dvh[dvh >= 50]) / len(dvh)
            case_scores['Bladder v50'] = min(max((1.0 - v50) / 0.5, 0.0), 1.0)
            # V60 Optimal 25% Mandatory 100%
            v60 = len(dvh[dvh >= 60]) / len(dvh)
            case_scores['Bladder v60'] = min(max((1.0 - v60) / 0.75, 0.0), 1.0)
            # V65 Optimal 10% Mandatory 50%
            v65 = len(dvh[dvh >= 65]) / len(dvh)
            case_scores['Bladder v65'] = min(max((0.5 - v65) / 0.4, 0.0), 1.0)
            # V70 Optimal 5% Mandatory 35%
            v70 = len(dvh[dvh >= 70]) / len(dvh)
            case_scores['Bladder v70'] = min(max((0.35 - v70) / 0.3, 0.0), 1.0)
            case_scores['Bladder'] = (case_scores['Bladder v50'] + case_scores['Bladder v60'] +
                                      case_scores['Bladder v65'] + 2 * case_scores['Bladder v70'])
            case_scores['OARs'] = case_scores['OARs'] + case_scores['Bladder']

            if output_dvh_values:
                case_scores['Bladder v50'] = v50
                case_scores['Bladder v60'] = v60
                case_scores['Bladder v65'] = v65
                case_scores['Bladder v70'] = v70
        else:
            case_scores['Bladder v50'] = 'n/a'
            case_scores['Bladder v60'] = 'n/a'
            case_scores['Bladder v65'] = 'n/a'
            case_scores['Bladder v70'] = 'n/a'
            case_scores['Bladder'] = 'n/a'

        # Femoral Head Left
        if 'Femur_Head_L' in organ_dvhs:
            number_of_oars = number_of_oars + 1
            dvh = organ_dvhs['Femur_Head_L']
            # V50 Optimal 5% Mandatory 25%
            v50 = len(dvh[dvh >= 50]) / len(dvh)
            case_scores['Femur_Head_L v50'] = min(max((0.25 - v50) / 0.2, 0.0), 1.0)
            case_scores['Femur_Head_L'] = 5 * case_scores['Femur_Head_L v50']
            case_scores['OARs'] = case_scores['OARs'] + case_scores['Femur_Head_L']

            if output_dvh_values:
                case_scores['Femur_Head_L v50'] = v50
        else:
            case_scores['Femur_Head_L v50'] = 'n/a'
            case_scores['Femur_Head_L'] = 'n/a'

        # Femoral Head Right
        if 'Femur_Head_R' in organ_dvhs:
            number_of_oars = number_of_oars + 1
            dvh = organ_dvhs['Femur_Head_R']
            # V50 Optimal 5% Mandatory 25%
            v50 = len(dvh[dvh >= 50]) / len(dvh)
            case_scores['Femur_Head_R v50'] = min(max((0.25 - v50) / 0.2, 0.0), 1.0)
            case_scores['Femur_Head_R'] = 5 * case_scores['Femur_Head_R v50']
            case_scores['OARs'] = case_scores['OARs'] + case_scores['Femur_Head_R']

            if output_dvh_values:
                case_scores['Femur_Head_R v50'] = v50
        else:
            case_scores['Femur_Head_R v50'] = 'n/a'
            case_scores['Femur_Head_R'] = 'n/a'

        # The final score should not depend on what is contoured. Either way the consensus will have all the OARs
        case_scores['OARs'] = case_scores['OARs'] * 10.0 / 5  # number_of_oars

        if case_number in [1, 4, 6, 8, 10, 14]:  # Prostate Only
            prescribed_dose = 74.0
            if 'CTV_Prostate' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_Prostate']
                d_min = np.min(dvh)
                case_scores['CTV_Prostate D_min'] = 1.0 - min(
                    max((prescribed_dose - d_min) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_Prostate D_min']
            else:
                case_scores['CTV_Prostate D_min'] = 'n/a'

            if 'CTV_SeminalVes' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_SeminalVes']
                d_min = np.min(dvh)
                case_scores['CTV_SeminalVes D_min'] = 1 - min(max((60 - d_min) * 100.0 /
                                                                  prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_SeminalVes D_min']
            else:
                case_scores['CTV_SeminalVes D_min'] = 'n/a'

            if 'PTVp_7400' in organ_dvhs:
                number_of_targets = number_of_targets + 3
                dvh = organ_dvhs['PTVp_7400']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                d_max = np.max(dvh)
                case_scores['PTVp_7400 D_median'] = 1.0 - min(
                    max(abs((prescribed_dose - d_median) * 100 / prescribed_dose / 1.0), 0.0), 1.0)
                # case_scores['PTVp_7400 D_min'] = 1.0 - min(max((prescribed_dose - d_min) * 100.0 /
                # prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_7400 D95'] = 1.0 - min(
                    max((prescribed_dose - d_95) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_7400 D_max'] = 1.0 - min(
                    max((d_max - prescribed_dose) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7400 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7400 D95']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7400 D_max']
            else:
                case_scores['PTVp_7400 D_median'] = 'n/a'
                case_scores['PTVp_7400 D95'] = 'n/a'
                case_scores['PTVp_7400 D_max'] = 'n/a'

            if 'PTVp_7100_dvh' in organ_dvhs:
                number_of_targets = number_of_targets + 2
                dvh = organ_dvhs['PTVp_7100_dvh']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                case_scores['PTVp_7100 D_median'] = 1 - min(max((71 - d_median) * 100.0 / prescribed_dose / 1.0,
                                                                0.0), 1.0)
                # case_scores['PTVp_7100 D_min'] = 1 - min(max((71 - d_min) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_7100 D95'] = 1 - min(max((71 - d_95) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7100 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7100 D95']
            else:
                case_scores['PTVp_7100 D_median'] = 'n/a'
                case_scores['PTVp_7100 D95'] = 'n/a'

            if 'PTVp_6000_dvh' in organ_dvhs:
                number_of_targets = number_of_targets + 2
                dvh = organ_dvhs['PTVp_6000_dvh']
                d_median = np.median(dvh)  # / 74.0 * 100.0
                # d_min = np.min(dvh)  # / 74.0 * 100.0
                d_95 = np.percentile(dvh, 95.0)
                case_scores['PTVp_6000 D_median'] = 1 - min(max((60 - d_median) * 100.0 / prescribed_dose / 1.0,
                                                                0.0), 1.0)
                # case_scores['PTVp_6000 D_min'] = 1 - min(max((60 - d_min) * 100.0 / prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['PTVp_6000 D95'] = 1 - min(max((60 - d_95) * 100.0 / prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6000 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6000 D95']
            else:
                case_scores['PTVp_6000 D_median'] = 'n/a'
                case_scores['PTVp_6000 D95'] = 'n/a'

            # The final score should not depend on what is contoured. Either way the consensus will have all the
            # target structures.
            case_scores['Targets'] = case_scores['Targets'] * 50.0 / 9  # number_of_targets

            if output_dvh_values:
                dvh = organ_dvhs['PTVp_7400']
                case_scores['PTVp_7400 D_median'] = np.median(dvh)
                case_scores['PTVp_7400 D_min'] = np.min(dvh)
                case_scores['PTVp_7400 D_max'] = np.max(dvh)
                dvh = organ_dvhs['PTVp_7100_dvh']
                case_scores['PTVp_7100 D_median'] = np.median(dvh)
                case_scores['PTVp_7100 D_min'] = np.min(dvh)
                dvh = organ_dvhs['PTVp_6000_dvh']
                case_scores['PTVp_6000 D_median'] = np.median(dvh)
                case_scores['PTVp_6000 D_min'] = np.min(dvh)

        elif case_number in [2, 5, 7, 9, 12, 15]:  # Prostate + nodes
            prescribed_dose = 74.0
            if 'CTV_Prostate' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_Prostate']
                d_min = np.min(dvh)
                case_scores['CTV_Prostate D_min'] = 1.0 - min(
                    max((prescribed_dose - d_min) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_Prostate D_min']
            else:
                case_scores['CTV_Prostate D_min'] = 'n/a'

            if 'CTV_SeminalVes' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_SeminalVes']
                d_min = np.min(dvh)
                case_scores['CTV_SeminalVes D_min'] = 1 - min(max((60 - d_min) * 100.0 /
                                                                  prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_SeminalVes D_min']
            else:
                case_scores['CTV_SeminalVes D_min'] = 'n/a'

            if 'PTVp_7400' in organ_dvhs:
                number_of_targets = number_of_targets + 3
                dvh = organ_dvhs['PTVp_7400']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                d_max = np.max(dvh)
                case_scores['PTVp_7400 D_median'] = 1 - min(
                    max(abs((prescribed_dose - d_median) * 100.0 / prescribed_dose / 1.0), 0.0), 1.0)
                # case_scores['PTVp_7400 D_min'] = 1.0 - min(max((prescribed_dose - d_min) * 100.0 /
                # prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_7400 D95'] = 1.0 - min(
                    max((prescribed_dose - d_95) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_7400 D_max'] = 1.0 - min(
                    max((d_max - prescribed_dose) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7400 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7400 D95']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7400 D_max']
            else:
                case_scores['PTVp_7400 D_median'] = 'n/a'
                case_scores['PTVp_7400 D95'] = 'n/a'
                case_scores['PTVp_7400 D_max'] = 'n/a'

            if 'PTVp_7100_dvh' in organ_dvhs:
                number_of_targets = number_of_targets + 2
                dvh = organ_dvhs['PTVp_7100_dvh']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                case_scores['PTVp_7100 D_median'] = 1.0 - min(max((71.0 - d_median) * 100.0 / prescribed_dose / 1.0,
                                                                  0.0), 1.0)
                # case_scores['PTVp_7100 D_min'] = 1 - min(max((71 - d_min) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_7100 D95'] = 1 - min(max((71 - d_95) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7100 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_7100 D95']
            else:
                case_scores['PTVp_7100 D_median'] = 'n/a'
                case_scores['PTVp_7100 D95'] = 'n/a'

            if 'PTVp_6000_dvh' in organ_dvhs:
                number_of_targets = number_of_targets + 2
                dvh = organ_dvhs['PTVp_6000_dvh']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                case_scores['PTVp_6000 D_median'] = 1.0 - min(max((60.0 - d_median) * 100.0 / prescribed_dose /
                                                                  1.0, 0.0), 1.0)
                # case_scores['PTVp_6000 D_min'] = 1 - min(max((60 - d_min) * 100.0 / prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['PTVp_6000 D95'] = 1 - min(max((60 - d_95) * 100.0 / prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6000 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6000 D95']
            else:
                case_scores['PTVp_6000 D_median'] = 'n/a'
                case_scores['PTVp_6000 D95'] = 'n/a'

            if 'CTV_LN_Pelvic' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_LN_Pelvic']
                d_min = np.min(dvh)
                case_scores['CTV_LN_Pelvic D_min'] = 1.0 - min(max((60.0 - d_min) * 100.0 / prescribed_dose / 4.0, 0.0),
                                                         1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_LN_Pelvic D_min']
            else:
                case_scores['CTV_LN_Pelvic D_min'] = 'n/a'

            if 'PTVn_6000' in organ_dvhs:
                number_of_targets = number_of_targets + 2
                dvh = organ_dvhs['PTVn_6000']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                case_scores['PTVn_6000 D_median'] = 1.0 - min(
                    max((60.0 - d_median) * 100.0 / prescribed_dose / 1.0, 0.0), 1.0)
                # case_scores['PTVn_6000 D_min'] = 1.0 - min(max((60.0 - d_min) * 100.0 /
                # prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['PTVn_6000 D95'] = 1.0 - min(max((60.0 - d_95) * 100.0 / prescribed_dose / 4.0, 0.0),
                                                         1.0)
                case_scores['PTVn_6000 D_median_value'] = d_median
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVn_6000 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVn_6000 D95']
            else:
                case_scores['PTVn_6000 D_median'] = 'n/a'
                case_scores['PTVn_6000 D95'] = 'n/a'

            # The final score should not depend on what is contoured. Either way the consensus will have all the
            # target structures.
            case_scores['Targets'] = case_scores['Targets'] * 50.0 / 12  # number_of_targets

            if output_dvh_values:
                dvh = organ_dvhs['PTVp_7400']
                case_scores['PTVp_7400 D_median'] = np.median(dvh)
                case_scores['PTVp_7400 D_min'] = np.min(dvh)
                case_scores['PTVp_7400 D_max'] = np.max(dvh)
                dvh = organ_dvhs['PTVp_7100_dvh']
                case_scores['PTVp_7100 D_median'] = np.median(dvh)
                case_scores['PTVp_7100 D_min'] = np.min(dvh)
                dvh = organ_dvhs['PTVp_6000_dvh']
                case_scores['PTVp_6000 D_median'] = np.median(dvh)
                case_scores['PTVp_6000 D_min'] = np.min(dvh)
                if 'PTVn_6000' in organ_dvhs:
                    dvh = organ_dvhs['PTVn_6000']
                    case_scores['PTVn_6000 D_median'] = np.median(dvh)
                    case_scores['PTVn_6000 D_min'] = np.min(dvh)

        else:  # Prostate Bed + Nodes
            prescribed_dose = 66.0
            if 'CTV_ProstateBed' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_ProstateBed']
                d_min = np.min(dvh)
                case_scores['CTV_ProstateBED D_min'] = 1.0 - min(
                    max((prescribed_dose - d_min) * 100.0 / prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_ProstateBED D_min']
            else:
                case_scores['CTV_ProstateBed D_min'] = 'n/a'

            if 'PTVp_6600' in organ_dvhs:
                number_of_targets = number_of_targets + 3
                dvh = organ_dvhs['PTVp_6600']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                d_max = np.max(dvh)
                case_scores['PTVp_6600 D_median'] = 1.0 - min(
                    max(abs((prescribed_dose - d_median) * 100.0 / prescribed_dose / 1.0), 0.0), 1.0)
                # case_scores['PTVp_6600 D_min'] = 1.0 - min(max((prescribed_dose - d_min) * 100.0 /
                # prescribed_dose / 5.0, 0.0), 1.0)
                case_scores['PTVp_6600 D95'] = 1.0 - min(max((prescribed_dose - d_95) * 100.0 / prescribed_dose / 5.0,
                                                             0.0), 1.0)
                case_scores['PTVp_6600 D_max'] = 1.0 - min(
                    max((d_max - prescribed_dose) * 100.0 / prescribed_dose / 5.0,
                        0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6600 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6600 D95']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVp_6600 D_max']
            else:
                case_scores['PTVp_6600 D_median'] = 'n/a'
                case_scores['PTVp_6600 D95'] = 'n/a'
                case_scores['PTVp_6600 D_max'] = 'n/a'

            if 'CTV_LN_Pelvic' in organ_dvhs:
                number_of_targets = number_of_targets + 1
                dvh = organ_dvhs['CTV_LN_Pelvic']
                d_min = np.min(dvh)
                case_scores['CTV_LN_Pelvic D_min'] = 1.0 - min(max((50.0 - d_min) * 100.0 /
                                                                   prescribed_dose / 4.0, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['CTV_LN_Pelvic D_min']
            else:
                case_scores['CTV_LN_Pelvic D_min'] = 'n/a'

            if 'PTVn_5000' in organ_dvhs:
                number_of_targets = number_of_targets + 2
                dvh = organ_dvhs['PTVn_5000']
                d_median = np.median(dvh)
                # d_min = np.min(dvh)
                d_95 = np.percentile(dvh, 95.0)
                case_scores['PTVn_5000 D_median'] = 1 - min(max((50.0 - d_median) * 100.0 / prescribed_dose / 1.0, 0.0),
                                                            1.0)
                # case_scores['PTVn_5000 D_min'] = 1 - min(max((50.0 - d_min) * 100.0 /
                # prescribed_dose / 3.4, 0.0), 1.0)
                case_scores['PTVn_5000 D95'] = 1 - min(max((50.0 - d_95) * 100.0 / prescribed_dose / 3.4, 0.0), 1.0)
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVn_5000 D_median']
                case_scores['Targets'] = case_scores['Targets'] + case_scores['PTVn_5000 D95']
            else:
                case_scores['PTVn_5000 D_median'] = 'n/a'
                case_scores['PTVn_5000 D95'] = 'n/a'

            # The final score should not depend on what is contoured. Either way the consensus will have all the
            # target structures.
            case_scores['Targets'] = case_scores['Targets'] * 50.0 / 7  # number_of_targets

            if output_dvh_values:
                dvh = organ_dvhs['PTVp_6600']
                case_scores['PTVp_6600 D_median'] = np.median(dvh)
                case_scores['PTVp_6600 D_min'] = np.min(dvh)
                case_scores['PTVp_6600 D_max'] = np.max(dvh)
                if 'PTVn_5000' in organ_dvhs:
                    dvh = organ_dvhs['PTVn_5000']
                    case_scores['PTVn_5000 D_median'] = np.median(dvh)
                    case_scores['PTVn_5000 D_min'] = np.min(dvh)

        case_scores['Overall'] = (case_scores['Targets'] + case_scores['OARs'])

        # Get contour similarity scores
        if contour_set >= 0:  # i.e. we don't calculate the APL of the test set against itself
            test_rtstruct_file = self.test_cases[case_number]['rtstruct_file']
            rtstruct_file_choice = "rtstruct_file_" + str(contour_set)
            reference_rtstruct_file = self.ground_truth_cases[case_number][rtstruct_file_choice]

            if (reference_rtstruct_file != "") & (test_rtstruct_file != ""):
                consensus_contour_results = score_autocontours_lib.score_a_case(reference_rtstruct_file,
                                                                                test_rtstruct_file,
                                                                                use_roi_name=True,
                                                                                dictionary_file=self.dictionary)
                for contour_similarity_result in consensus_contour_results:
                    field_name = contour_similarity_result['RefOrgan'] + " NAPL"
                    case_scores[field_name] = contour_similarity_result['APL'] / contour_similarity_result['ref_length']

        return case_scores

    def load_ct(self, case_number):
        # This function will load a CT from the image_folder for the case_number given
        image_cosines = [0] * 6
        normal_to_planes = [0, 0, 0]
        voxel_img_data = None
        sitk_output = sITK.Image()

        try:
            # Find all the files in the CT folder
            # and read the image slices
            paths_dcms = []
            slices = []
            for case in self.ground_truth_cases.values():
                if case['number'] == case_number:
                    ct_directory = self.ground_truth_cases[case_number]['image_folder']
                    for directory_item in os.listdir(ct_directory):
                        full_item_path = os.path.join(ct_directory, directory_item)
                        if os.path.isfile(full_item_path):
                            try:
                                ds = pydcm.filereader.dcmread(full_item_path)
                                slices.append(ds)
                                paths_dcms.append(full_item_path)
                            except pydcm.errors.InvalidDicomError:
                                print('Skipping non dicom file: ', full_item_path)
                                continue

            # Sort the slices according to the normals
            if len(slices):
                # discard slices without an ImagePositionPatient
                slices = list(filter(lambda x: 'ImagePositionPatient' in x, slices))
                image_cosines = slices[0].ImageOrientationPatient
                normal_to_planes = np.cross(image_cosines[0:3], image_cosines[3:6])
                for i in range(0, len(slices)):
                    image_position_patient = slices[i].ImagePositionPatient
                    normal_position = np.dot(image_position_patient, normal_to_planes)
                    # Abusing the slice location tag
                    slices[i].SliceLocation = normal_position
                slices.sort(key=lambda x: float(x.SliceLocation))

                # Stack the image data into a voxel array
                slices_data = []
                for s_id, s in enumerate(slices):
                    try:
                        real_world_data = pydcm.pixel_data_handlers.apply_rescale(s.pixel_array, s)
                        slices_data.append(real_world_data)
                    except:
                        pass
                if len(slices_data):
                    voxel_img_data = np.stack(slices_data, axis=0)  # [row, col, plane], [D,H,W] or (z,y,x)

            # Extract the other geometry parameters
            if len(slices):
                spacing = np.array(slices[0].PixelSpacing).tolist() + [float(slices[0].SliceThickness)]
                origin = slices[0].ImagePositionPatient
                slice_axis_positions = [s.SliceLocation for s in slices]

                # Store as a sitk image
                # Although we are using ITK for storage, we are not using it for loading the DICOM
                # to give us more control over the slice spacing
                sitk_output = sITK.GetImageFromArray(voxel_img_data)  # SimpleITK reorders to x,y,z
                sitk_output.SetOrigin(origin)
                sitk_output.SetSpacing(spacing)
                sitk_output.SetDirection([image_cosines[0],
                                          image_cosines[1],
                                          image_cosines[2],
                                          image_cosines[3],
                                          image_cosines[4],
                                          image_cosines[5],
                                          normal_to_planes[0],
                                          normal_to_planes[1],
                                          normal_to_planes[2]])

        except:
            traceback.print_exc()

        if True:
            print(' Image loaded\n   Size: ', sitk_output.GetSize())
            print('   Spacing: ', sitk_output.GetSpacing())
            print('   Origin : ', sitk_output.GetOrigin())
            print('')

        return sitk_output

    def load_dose(self, case_number):
        sitk_output = sITK.Image()

        try:
            # Dose file is a single file.

            dose_file = ""
            for case in self.test_cases.values():
                if case['number'] == case_number:
                    # Dose file is a single file.
                    dose_file = self.test_cases[case_number]['dose_file']

            # Get the slice thickness?
            if dose_file != "":
                try:
                    ds = pydcm.filereader.dcmread(dose_file)
                except:
                    print('Could not read DOSE file: ', dose_file)
                    return
                # D,H,W / z,y,x
                dose_volume_data = pydcm.pixel_data_handlers.apply_rescale(ds.pixel_array, ds) * ds.DoseGridScaling
                dose_dims = [ds.Rows, ds.Columns, int(ds.NumberOfFrames)]

                if (ds.SliceThickness is not None) & (ds.SliceThickness != 0):
                    spacing = np.array(ds.PixelSpacing).tolist() + [float(ds.SliceThickness)]
                else:
                    slice_positions = np.array(ds.GridFrameOffsetVector)
                    slice_differences = np.ediff1d(slice_positions)
                    slice_thickness = statistics.mode(slice_differences)
                    spacing = np.array(ds.PixelSpacing).tolist() + [float(slice_thickness)]
                origin = ds.ImagePositionPatient
                image_cosines = ds.ImageOrientationPatient
                normal_to_planes = np.cross(image_cosines[0:3], image_cosines[3:6])

                # Store as a sitk image
                # sITK reorders to x,y,z
                sitk_output = sITK.GetImageFromArray(dose_volume_data)
                sitk_output.SetOrigin(origin)
                sitk_output.SetSpacing(spacing)
                sitk_output.SetDirection([image_cosines[0],
                                          image_cosines[1],
                                          image_cosines[2],
                                          image_cosines[3],
                                          image_cosines[4],
                                          image_cosines[5],
                                          normal_to_planes[0],
                                          normal_to_planes[1],
                                          normal_to_planes[2]])

        except:
            traceback.print_exc()

        if True:
            print(' Dose loaded\n   Size: ', sitk_output.GetSize())
            print('   Spacing: ', sitk_output.GetSpacing())
            print('   Origin : ', sitk_output.GetOrigin())
            print('')

        return sitk_output

    def load_rtstructs(self, case_number, image_volume, required_labels, test=False, expert=0, rename=True):
        contours_list = []
        label_map = {}

        # RTSS file is a single file.
        rtstruct_file = ""
        if test:
            for case in self.test_cases.values():
                if case['number'] == case_number:
                    # RTStruct file is a single file.
                    rtstruct_file = self.test_cases[case_number]['rtstruct_file']
        else:
            for case in self.ground_truth_cases.values():
                if case['number'] == case_number:
                    # RTStruct file is a single file.
                    rtstruct_file_choice = "rtstruct_file_" + str(expert)
                    rtstruct_file = self.ground_truth_cases[case_number][rtstruct_file_choice]

        rtstruct_data = pydcm.filereader.dcmread(rtstruct_file)

        # Rename the structures using the dictionary so that we can process them more robustly
        # They should be correct anyway since they are our ground truth, but "just in case"
        if rename:
            rtstruct_data = rename_structures(rtstruct_data, self.dictionary)

        # Extract all different contours
        for i in range(len(rtstruct_data.ROIContourSequence)):
            # Get contour
            contour_obj = {'color': list(rtstruct_data.ROIContourSequence[i].ROIDisplayColor)}
            has_contours = False
            if hasattr(rtstruct_data.ROIContourSequence[i], "ContourSequence"):
                contour_obj['contours'] = [s.ContourData for s in rtstruct_data.ROIContourSequence[i].ContourSequence]
                has_contours = True
            contour_obj['number'] = rtstruct_data.ROIContourSequence[i].ReferencedROINumber
            for j in range(len(rtstruct_data.StructureSetROISequence)):
                if rtstruct_data.StructureSetROISequence[j].ROINumber == contour_obj['number']:
                    contour_obj['name'] = str(rtstruct_data.StructureSetROISequence[j].ROIName)

            # Remove items not in the required_labels
            if (contour_obj['name'] in required_labels.keys()) & has_contours:
                contours_list.append(contour_obj)
                label_map[contour_obj['number']] = contour_obj['name']

        # Order your contours
        if len(contours_list):
            contours_list = list(sorted(contours_list, key=lambda obj: obj['number']))

        # Raster into a volume
        # z = self.z_vals  # we didn't store that when we read the volume
        pos_r = image_volume.GetOrigin()[0]  # origin[1]
        spacing_r = image_volume.GetSpacing()[0]  # self.spacing[1]
        pos_c = image_volume.GetOrigin()[1]  # self.origin[0]
        spacing_c = image_volume.GetSpacing()[1]  # self.spacing[0]
        vol_size = image_volume.GetSize()
        # structure_mask_data = np.zeros([vol_size[0], vol_size[1], vol_size[2], number_of_labels], dtype=np.uint8)
        normal_to_planes = np.array(image_volume.GetDirection()[6:9])
        slice_axis_positions = np.zeros(image_volume.GetSize()[2])
        image_origin = np.array(image_volume.GetOrigin())
        slice_thickness = image_volume.GetSpacing()[2]
        for i in range(0, image_volume.GetSize()[2]):
            slice_origin = image_origin + i * normal_to_planes * slice_thickness
            slice_axis_positions[i] = np.dot(slice_origin, normal_to_planes)
        slice_axis_positions = slice_axis_positions.tolist()

        # Array of SITK objects
        segmentations = {}

        if len(contours_list):
            for contour_obj in contours_list:
                try:
                    # if len(required_labels):
                    #    class_id = int(required_labels.get(contour_obj['name'], 0))
                    # else:
                    #    class_id = int(contour_obj['number'])

                    print('Loading: {},'.format(contour_obj['name']), end='')

                    # Load as z,y,x to be consistent with image behaviour
                    structure_mask_data = np.zeros([vol_size[2], vol_size[1], vol_size[1]], dtype=np.uint8)

                    min_z = 9999999999
                    max_z = -9999999999
                    for c_id, contour in enumerate(contour_obj['contours']):
                        coords = np.array(contour).reshape((-1, 3))

                        if len(coords) > 1:
                            normal_positions = np.zeros(coords.shape[0])
                            for i in range(0, coords.shape[0]):
                                normal_positions[i] = np.dot(coords[i, 0:3], normal_to_planes)

                            assert np.amax(np.abs(np.diff(normal_positions[:]))) == 0
                            # Get the nearest slice index for this contour
                            # They should match exactly, but there are precision issues
                            slice_index = \
                                min(enumerate(slice_axis_positions), key=lambda x: abs(x[1] - normal_positions[0]))[0]
                            if slice_index < min_z:
                                min_z = slice_index
                            if slice_index > max_z:
                                max_z = slice_index

                            # Raster
                            rows = (coords[:, 1] - pos_r) / spacing_r  # pixel_idx = f(real_world_idx, ct_resolution)
                            cols = (coords[:, 0] - pos_c) / spacing_c
                            r_index, c_index = skimage.draw.polygon(rows, cols)  # rr --> y-axis, cc --> x-axis
                            # structure_mask_data[ slice_index, r_index, c_index, class_id] = 1
                            # This would do everything in a single mask, so nothing can overlap,
                            # but that's not the case, so we need one mask per object

                            structure_mask_data[slice_index, r_index, c_index] = 1

                    # print(' (voxel-count=)', len(structure_mask_data[structure_mask_data[:, :, :, class_id] == 1]))
                    print(' number of voxels: ', len(structure_mask_data[structure_mask_data[:, :, :] == 1]))

                    # Let sITK reorder to x,y,z
                    segmentation = sITK.GetImageFromArray(structure_mask_data)
                    segmentation.SetOrigin(image_volume.GetOrigin())
                    segmentation.SetSpacing(image_volume.GetSpacing())
                    segmentation.SetDirection(image_volume.GetDirection())
                    segmentation.SetMetaData("Structure Name", contour_obj['name'])
                    segmentations[contour_obj['name']] = segmentation

                except:
                    print(" - Error converting ", contour_obj['name'], " to a voxel array")
                    traceback.print_exc()

        return segmentations
