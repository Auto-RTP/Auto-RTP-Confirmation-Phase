#
# original code taken from score_autocontours.py from
# https://github.com/Auto-segmentation-in-Radiation-Oncology/Chapter-15
# by Mark Gooding 2020  Mirada Medical Ltd
#
# Revision
# Djamal Boukerroui, 11-2022
# Minor changes + adding all asymmetric measures to output.
# A few upgrades to newer versions of packages
#
# Revision
# Mark Gooding, 05-2023
# Modified for AUTO-RTP challenge
# - Added ability to use a dictionary
# - Removed distance measure calculation for speed

import pydicom
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
from shapely.ops import unary_union
from shapely.ops import split
import numpy as np
from scipy import stats as spstats
import random
import json


def load_dictionary_of_structure(structure_dictionary_filename):
    with open(structure_dictionary_filename) as f:
        structure_dictionary = json.load(f)

    # TODO check for invalid structure / lack of information

    return structure_dictionary


def find_structure_match(ref_roi_name, test_structure_data, structure_dictionary):
    # Find matching structures according to the dictionary

    test_match_name = "No match"
    test_match_number = -1
    ref_roi_name_from_dictionary = ref_roi_name
    for dict_key, dict_variants in structure_dictionary.items():
        if ref_roi_name.lower().strip() == dict_key.lower().strip():
            ref_roi_name_from_dictionary = dict_key
            break
        else:
            for variant in dict_variants:
                if ref_roi_name.lower().strip() == variant.lower().strip():
                    ref_roi_name_from_dictionary = dict_key
                    break

    for idx, test_roi in enumerate(test_structure_data.StructureSetROISequence):
        test_roi_name = test_roi.ROIName
        test_roi_name_from_dictionary = test_roi_name
        for dict_key, dict_variants in structure_dictionary.items():
            if test_roi_name.lower().strip() == dict_key.lower().strip():
                test_roi_name_from_dictionary = dict_key
                break
            else:
                for variant in dict_variants:
                    if test_roi_name.lower().strip() == variant.lower().strip():
                        test_roi_name_from_dictionary = dict_key
                        break
        if test_roi_name_from_dictionary == ref_roi_name_from_dictionary:
            test_match_name = test_structure_data.StructureSetROISequence[idx].ROIName
            test_match_number = test_structure_data.StructureSetROISequence[idx].ROINumber
            break

    return test_match_name, test_match_number


def get_distance_measures(ref_poly, test_poly, stepsize=1.0, warningsize=1.0):
    """
    Compute distance scores Hausdorff, 95%HD, Mean and Median Distances

    :param ref_poly:  Reference polygon
    :param test_poly: Test Polygon
    :param stepsize:  Stepsize is a arc length fro integration along the curve
    :param warningsize:
    :return: a list of distances along the polygons ref_to_test, test_2_ref
    """
    # Hausdorff is trivial to compute with Shapely, but average distance requires stepping along each polygon.
    # This is the 'stepsize' in mm. At each point the minimum distance to the other contour is calculated to
    # create a list of distances. From this list the HD can be calculated from this, but it is inaccurate. Therefore,
    # we compare it to the Shapely one and report a problem if the error is greater that 'warningsize' in mm.

    reference_line = ref_poly.boundary
    test_line = test_poly.boundary

    distance_ref_to_test = []
    for distance_along_contour in np.arange(0, reference_line.length, stepsize):
        distance_to_other = reference_line.interpolate(distance_along_contour).distance(test_line)
        distance_ref_to_test.append(distance_to_other)

    distance_test_to_ref = []
    for distance_along_contour in np.arange(0, test_line.length, stepsize):
        distance_to_other = test_line.interpolate(distance_along_contour).distance(reference_line)
        distance_test_to_ref.append(distance_to_other)

    my_hd = np.max([np.max(distance_ref_to_test), np.max(distance_test_to_ref)])
    shapely_hd = test_poly.hausdorff_distance(ref_poly)

    if abs(my_hd - shapely_hd) > warningsize: #TODO logging
        print('There is a discrepancy between the Hausdorff distance and the list used to calculate the 95% HD')
        print('You may wish to consider a smaller stepsize')

    return distance_ref_to_test, distance_test_to_ref


def get_added_path_length(ref_poly, contracted_poly, expanded_poly, debug=False):
    """
    Compute added path length

    :param ref_poly: Reference polygon
    :param contracted_poly:  contracted test polygon
    :param expanded_poly:   expanded test polygon
    :param debug:
    :return:
    """

    added_path_length = 0

    reference_boundary = ref_poly.boundary
    if contracted_poly.area > 0:
        contracted_boundary = contracted_poly.boundary
    else:
        contracted_boundary = None
    expanded_boundary = expanded_poly.boundary

    if contracted_boundary is not None:
        split_success = False
        split_attempts = 0
        while (not split_success) & (split_attempts < 5):
            try:
                ref_split_inside = split(reference_boundary, contracted_boundary)
                split_success = True
            except ValueError:
                # Error can occur if sections parallel. Try a small jitter?
                contracted_poly_new = unary_union(contracted_poly.buffer(random.random() * 0.0001, 32, 1, 1))
                contracted_boundary = contracted_poly_new.boundary
                split_attempts = split_attempts + 1

        if split_success:
            for line_segment in ref_split_inside.geoms:
                # check it the centre of the line is within the contracted polygon
                mid_point = line_segment.interpolate(0.5, True)
                if contracted_poly.contains(mid_point):
                    added_path_length = added_path_length + line_segment.length
        else:
            if debug:
                print('Failed to correctly calculate Added Path Length for a slice of an organ', flush=True)
                # would be nice if we had the information to return here!

    split_success = False
    split_attempts = 0
    while (not split_success) & (split_attempts < 5):
        try:
            ref_split_outside = split(reference_boundary, expanded_boundary)
            split_success = True
        except ValueError:
            # Error can occur if sections parallel. Try a tiny random jitter
            expanded_poly_new = unary_union(expanded_poly.buffer(random.random() * 0.0001, 32, 1, 1))
            expanded_boundary = expanded_poly_new.boundary
            split_attempts = split_attempts + 1

    if split_success:
        for line_segment in ref_split_outside.geoms:
            # check it the centre of the line is outside the expanded polygon
            mid_point = line_segment.interpolate(0.5, True)
            if not expanded_poly.contains(mid_point):
                added_path_length = added_path_length + line_segment.length
    else:
        if debug:
            print('Failed to correctly calculate Added Path Length for a slice of an organ', flush=True)
            # would be nice if we had the information to return here!

    return added_path_length


def find_and_score_slice_matches(ref_rtss, test_rtss, slice_thickness, contour_matches, tolerance=1.0, debug=False):
    """
    main function to compute the contour scores

    :param ref_rtss:
    :param test_rtss:
    :param slice_thickness:
    :param contour_matches:
    :param tolerance:
    :param debug:
    :return: a list of tulpes (refname, testname, scores)
    """
    result_list = []

    # for each structure if match list
    for idx, match_ids in enumerate(contour_matches):

        ref_id, test_id, ref_name, test_name = match_ids
        total_added_path_length = 0
        test_contour_length = 0
        ref_contour_length = 0
        total_true_positive_area = 0
        total_false_positive_area = 0
        total_false_negative_area = 0
        total_test_area = 0
        total_ref_area = 0
        # distance_ref_to_test = []
        # distance_test_to_ref = []
        ref_weighted_centroid_sum = np.array([0, 0, 0])
        test_weighted_centroid_sum = np.array([0, 0, 0])

        if debug:
            print(f'Computing scores for Reference: {ref_name} and Test {test_name})')

        ref_contour_set = None
        test_contour_set = None

        # Find contour set for reference and test
        for contour_set in ref_rtss.ROIContourSequence:
            if contour_set.ReferencedROINumber == ref_id:
                ref_contour_set = contour_set
                break
        for contour_set in test_rtss.ROIContourSequence:
            if contour_set.ReferencedROINumber == test_id:
                test_contour_set = contour_set
                break

        ref_polygon_dictionary, ref_z_slices = build_polygon_dictionary(ref_contour_set)
        test_polygon_dictionary, test_z_slices = build_polygon_dictionary(test_contour_set)

        # for each slice in ref find corresponding slice in test
        for z_value, refpolygon in ref_polygon_dictionary.items():
            if z_value in test_z_slices:
                testpolygon = test_polygon_dictionary[z_value]
                # go get some distance measures
                # these get added to a big list so that we can calculate the 95% HD
                # [ref_to_test, test_to_ref] = get_distance_measures(refpolygon, testpolygon, 0.05)
                # distance_ref_to_test.extend(ref_to_test)
                # distance_test_to_ref.extend(test_to_ref)

                # apply tolerance ring margin to test with added path length
                expanded_poly = unary_union(testpolygon.buffer(tolerance, 32, 1, 1))
                contracted_poly = unary_union(testpolygon.buffer(-tolerance, 32, 1, 1))

                # add intersection of contours
                contour_intersection = refpolygon.intersection(testpolygon)
                total_true_positive_area = total_true_positive_area + contour_intersection.area
                total_false_negative_area = total_false_negative_area + \
                                            (refpolygon.difference(contour_intersection)).area
                total_false_positive_area = total_false_positive_area + \
                                            (testpolygon.difference(contour_intersection)).area
                total_test_area = total_test_area + testpolygon.area
                total_ref_area = total_ref_area + refpolygon.area
                centroid_point = refpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)
                centroid_point = testpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

                # add length of remain contours
                added_path = get_added_path_length(refpolygon, contracted_poly, expanded_poly, debug=debug)
                total_added_path_length = total_added_path_length + added_path
                test_contour_length = test_contour_length + testpolygon.length
                ref_contour_length = ref_contour_length + refpolygon.length

            else:
                # if no corresponding slice, then add the whole ref length
                # print('Adding path for whole contour')
                path_length = refpolygon.length
                total_added_path_length = total_added_path_length + path_length
                ref_contour_length = ref_contour_length + refpolygon.length
                # also the whole slice is false negative
                total_false_negative_area = total_false_negative_area + refpolygon.area
                total_ref_area = total_ref_area + refpolygon.area
                centroid_point = refpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                ref_weighted_centroid_sum = ref_weighted_centroid_sum + (refpolygon.area * centroid_point_np)

        # we also need to consider the slices for which there is a test contour but no reference
        for z_value, testpolygon in test_polygon_dictionary.items():
            if z_value not in ref_z_slices:
                # add path length doesn't get updated
                test_contour_length = test_contour_length + testpolygon.length
                # but the whole slice is false positive
                total_false_positive_area = total_false_positive_area + testpolygon.area
                total_test_area = total_test_area + testpolygon.area
                centroid_point = testpolygon.centroid
                centroid_point_np = np.array([centroid_point.x, centroid_point.y, z_value])
                test_weighted_centroid_sum = test_weighted_centroid_sum + (testpolygon.area * centroid_point_np)

        # now we need to deal with the distance lists to work out the various distance measures
        # NOTE: these are different calculations to those used in plastimatch. The book chapter will explain all..

        # Added the test to avoid division bt zeros for empty structures.  We should have avoid it from happening before
        if total_ref_area > 0:
            ref_centroid = ref_weighted_centroid_sum / total_ref_area
        else:
            ref_centroid = np.array([0, 0, 0])
        if total_test_area > 0:
            test_centroid = test_weighted_centroid_sum / total_test_area
        else:
            test_centroid = np.array([0, 0, 0])

        # if distance_ref_to_test == [] and distance_test_to_ref == []:
        #     if debug:
        #         print('Empty contours or are not on the same slices!')
        #     hd_ref_to_test = hd_test_to_ref =  float('nan')
        #     hd95_ref_to_test = hd95_test_to_ref = float('nan')
        #     ave_dist_ref_to_test = ave_dist_test_to_ref = float('nan')
        #     hd50_dist_ref_to_test = hd50_dist_test_to_ref = float('nan')  # median
        # else:
        #     hd_ref_to_test = np.max(distance_ref_to_test)
        #     hd_test_to_ref = np.max(distance_test_to_ref)
        #     hd95_ref_to_test = np.percentile(distance_ref_to_test, 95)
        #     hd95_test_to_ref = np.percentile(distance_test_to_ref, 95)
        #     ave_dist_ref_to_test = np.mean(distance_ref_to_test)
        #     ave_dist_test_to_ref = np.mean(distance_test_to_ref)
        #     hd50_dist_ref_to_test = np.median(distance_ref_to_test)
        #     hd50_dist_test_to_ref = np.median(distance_test_to_ref)

        # print('added path length = ', total_added_path_length)
        result_list.append((ref_name, test_name,
                            [total_added_path_length, ref_contour_length, test_contour_length,
                             total_true_positive_area * slice_thickness,
                             total_false_negative_area * slice_thickness,
                             total_false_positive_area * slice_thickness, total_ref_area * slice_thickness,
                             total_test_area * slice_thickness,
                             # hd_ref_to_test, hd_test_to_ref,
                             # hd95_ref_to_test, hd95_test_to_ref,
                             # ave_dist_ref_to_test, ave_dist_test_to_ref,
                             # hd50_dist_ref_to_test, hd50_dist_test_to_ref,
                             ref_centroid, test_centroid]))

    return result_list


def build_polygon_dictionary(contour_set):
    # this function extracts the polygon data from the pydicom structure and
    # converts it to a dictionary of Shapely polygons by the z slice
    # TODO: Ideally this function could be improved for off-axis data.

    polygon_dictionary = {}
    z_slices = []
    if contour_set is not None:
        # get the list of z-values for the reference set
        for contour_slice in contour_set.ContourSequence:
            number_pts = int(contour_slice.NumberOfContourPoints)
            if number_pts >= 3:  # We check for zero volume/level at the slice level
                contour_points = contour_slice.ContourData
                z_slices.append(contour_points[2])
        # round to 1 decimal place (0.1mm) to make finding a match more robust
        z_slices = np.round(z_slices, 1)
        z_slices = np.unique(z_slices)

        # now build the multi-polygon for each z-slice
        for z_value in z_slices:
            polygon_data = None
            for contour_slice in contour_set.ContourSequence:
                number_pts = int(contour_slice.NumberOfContourPoints)
                if number_pts >= 3:
                    contour_points = contour_slice.ContourData
                    if np.round(contour_points[2], 1) == z_value:
                        # make 2D contours
                        contour_points_2d = np.zeros((number_pts, 2))
                        for i in range(0, number_pts):
                            contour_points_2d[i][0] = float(contour_points[i * 3])
                            contour_points_2d[i][1] = float(contour_points[i * 3 + 1])
                        if polygon_data is None:
                            # Make points into Polygon
                            polygon_data = Polygon(LinearRing(contour_points_2d))
                        else:
                            # Turn next set of points into a Polygon
                            current_polygon = Polygon(LinearRing(contour_points_2d))
                            # Attempt to fix any self-intersections in the resulting polygon
                            if not current_polygon.is_valid:
                                current_polygon = current_polygon.buffer(0)
                            if polygon_data.contains(current_polygon):
                                # if the new polygon is inside the old one, chop it out
                                polygon_data = polygon_data.difference(current_polygon)
                            elif polygon_data.within(current_polygon):
                                # if the new and vice versa
                                polygon_data = current_polygon.difference(polygon_data)
                            else:
                                # otherwise it is a floating blob to add
                                polygon_data = polygon_data.union(current_polygon)
                        # Attempt to fix any self-intersections in the resulting polygon
                        if polygon_data is not None:
                            if not polygon_data.is_valid:
                                polygon_data = polygon_data.buffer(0)
            # check this slice has a tangible size polygon.
            if (polygon_data.length > 0) & (polygon_data.area > 0):
                polygon_dictionary[z_value] = polygon_data
    return polygon_dictionary, z_slices


def estimate_slice_thickness(contour_data_set, debug=False):
    """
    Estimate the slice thickness without the CT image
    this is a crude attempt to estimate the slice thickness without loading the image
    we assume that the slices are equally spaced, and if we collect unique slice positions
    for enough slices with contours then the modal difference will represent the slice thickness

    :param contour_data_set:
    :param debug:  [False]
    :return: slice thickness
    """

    z_list = []
    z_diff_list = []

    for contour_set in contour_data_set.ROIContourSequence:
        if hasattr(contour_set, 'ContourSequence'):
            for contour_slice in contour_set.ContourSequence:
                contour_points = contour_slice.ContourData
                z_list.append(contour_points[2])

    z_list = np.unique(z_list)
    z_list = np.sort(z_list)

    old_z_val = z_list[0]
    for z_val in z_list:
        z_diff = z_val - old_z_val
        old_z_val = z_val
        z_diff_list.append(z_diff)

    slice_thickness = spstats.mode(z_diff_list, keepdims=False).mode
    if debug:
        print('slice thickness: ', slice_thickness)

    return slice_thickness


def score_a_case(reference_rtss_filename, test_rtss_filename, slice_thickness=0,
               use_roi_name=False, verbose=False, tolerance=1.0, dictionary_file=''):
    """

    :param reference_rtss_filename:
    :param test_rtss_filename:
    :param slice_thickness:
    :param output_filename:
    :param use_roi_name:  use roi name for filtering, if True match case insensitive
    :param verbose:
    :param tolerance:  APL tolerance
    :return:
    """
    # load the DICOM files
    ref_data = pydicom.read_file(reference_rtss_filename, False)
    test_data = pydicom.read_file(test_rtss_filename, False)

    use_dictionary = False
    if dictionary_file != '':
        structure_dictionary = load_dictionary_of_structure(dictionary_file)
        use_dictionary = True

    if slice_thickness == 0:
        slice_thickness = estimate_slice_thickness(ref_data, debug=verbose)

    nb_ref_rois = len(ref_data.StructureSetROISequence)
    nb_test_rois = len(test_data.StructureSetROISequence)

    if nb_ref_rois > 1 or nb_test_rois > 1:
        use_roi_name = True
    # find the matching structure names if needed
    contour_matches = []
    for ref_roi in ref_data.StructureSetROISequence:
        ref_name = ref_roi.ROIName
        ref_id = ref_roi.ROINumber
        ref_contour_set = None

        # Find contour set for reference
        for contour_set in ref_data.ROIContourSequence:
            if contour_set.ReferencedROINumber == ref_id:
                ref_contour_set = contour_set
                break

        # Don't bother checking for a match if the reference is empty
        # TODO could also check for 0 length or 0 volume, but that is more effort
        if ref_contour_set is None:
            print_verbose(verbose, 'Reference contour is empty for structure', ref_name)
            continue

        if hasattr(ref_contour_set, 'ContourSequence'):
            number_of_matches = 0
            print_verbose(verbose, 'Searching for matching structures for: {:s}'.format(ref_name))

            if use_roi_name:
                matched_roi_number = -1
                if use_dictionary:
                    matched_roi_name, matched_roi_number = find_structure_match(ref_name, test_data,
                                                                                structure_dictionary)
                    if matched_roi_number != -1:
                        number_of_matches = number_of_matches + 1
                        match_data = (ref_id, matched_roi_number, ref_name, matched_roi_name)
                if matched_roi_number == -1:
                    for test_roi in test_data.StructureSetROISequence:
                        test_name = test_roi.ROIName

                        # else we just use the name given
                        if test_name.lower().strip() == ref_name.lower().strip():
                            number_of_matches = number_of_matches + 1
                            match_data = (ref_id, test_roi.ROINumber, ref_name, test_name)

                if number_of_matches == 1:
                    contour_matches.append(match_data)
                elif number_of_matches == 0:
                    if verbose:
                        print_verbose(verbose, 'No match for structure: {:s}, Skipping structure'.format(ref_name))
                elif number_of_matches > 1:
                    # TODO compare to each and report for all?
                    print_verbose(verbose, 'Multiple matches for structure: {:s}, Skipping structure'.format(ref_name))
            else:
                # Expect one ROI :  Check the ROI has a ContourSequence
                if hasattr(test_data.ROIContourSequence[0], 'ContourSequence'):
                    test_name = test_data.StructureSetROISequence[0].ROIName
                    test_id = test_data.StructureSetROISequence[0].ROINumber
                    match_data = (ref_id, test_id, ref_name, test_name)
                    contour_matches.append(match_data)

    resultlist = find_and_score_slice_matches(ref_data, test_data, slice_thickness, contour_matches,
                                              tolerance=tolerance, debug=verbose)

    auto_contour_measures = format_result_list(resultlist)
    return auto_contour_measures


def print_verbose(verbose, *args):
    if verbose:
        print(*args)

def format_result_list(result_list):
    formatted_results = []
    for result in result_list:
        ref_name, test_name, scores = result
        # scores[0] APL
        # scores[1] Ref PL
        # scores[2] Test PL
        # scores[3] TP volume
        # scores[4] FN volume
        # scores[5] FP volume
        # scores[6] Ref volume
        # scores[7] Test volume
        # scores[8] HD ref2test
        # scores[9] HD test2ref
        # scores[10] HD95 ref2test
        # scores[11] HD95 test2ref
        # scores[12] Average Distance  ref2test
        # scores[13] Average Distance  test2ref
        # scores[14] Median Distance   ref2test
        # scores[15] Median Distance   test2ref
        # scores[16] Reference Centroid
        # scores[17] Test Centroid
        results_structure = {'RefOrgan': ref_name, 'TestOrgan': test_name,
                             'APL': scores[0], 'ref_length': scores[1], 'test_length': scores[2],
                             'TPVol': scores[3], 'FNVol': scores[4],
                             'FPVol': scores[5], 'RefVol': scores[6], 'TestVol': scores[7],
                             'SEN': scores[3] / scores[6], 'FPFrac': scores[5] / scores[6],
                             'Inclusiveness': scores[3] / scores[7], 'PPV': scores[3]/(scores[3]+scores[5]),
                             'three_D_DSC': 2 * scores[3] / (scores[6] + scores[7]),
                             'ref_cent': scores[8], 'test_cent': scores[9],
                             'cent_dist': np.linalg.norm(scores[8] - scores[9])}
        formatted_results.append(results_structure)
    return formatted_results


