from Autortp import Autortp

if __name__ == "__main__":
    evaluator = Autortp()
    if evaluator.validate() > 0:
        exit(1)
    print("Validation successful")
    if evaluator.evaluate() > 0:
        exit(1)
    print("Finished scoring")
