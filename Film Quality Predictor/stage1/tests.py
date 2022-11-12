from hstest import *
from math import isclose


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


class Stage1Test(StageTest):

    def check_outputs_number(self, values_number: int, user_output: str):
        outputs = user_output.split()

        if not all(is_float(output) for output in outputs):
            raise WrongAnswer(f"Answer '{user_output}' contains non-numeric values.")

        if len(outputs) != values_number:
            raise WrongAnswer(f"Answer contains {len(outputs)} values, but {values_number} values are expected.")

    def check_num_values(self, values: list, user_values: list, message: str, rel_tol=1.0e-2):
        if not all(isclose(value, user_value, rel_tol=rel_tol) for value, user_value in zip(values, user_values)):
            raise WrongAnswer(message)

    @dynamic_test
    def test(self):
        pr = TestedProgram()
        user_output = pr.start().strip()

        if len(user_output.strip()) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        # check output format
        self.check_outputs_number(3, user_output)

        # check number of rows
        answer = [32086]
        user_values = [float(value) for value in user_output.split()][:1]
        self.check_num_values(answer, user_values,
                              "The number of rows in the dataset is wrong.\n"
                              "Check how you read the header.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=0)

        # check number of rows after filtering
        answer = [25000]
        user_values = [float(value) for value in user_output.split()][1:2]
        self.check_num_values(answer, user_values,
                              "The number of rows in the dataset after filtering is wrong.\n"
                              "Check the conditions of filtering.\n"
                              "Make sure that you provide numbers in the correct order.",
                              rel_tol=0)

        # check proportions of classes
        answer = [0.5]
        user_values = [float(value) for value in user_output.split()][2:3]
        self.check_num_values(answer, user_values, 'The proportion of "good" films is wrong.\n'
                              "Make sure that you provide numbers in the correct order.", rel_tol=1.0e-2)

        return CheckResult.correct()


if __name__ == '__main__':
    Stage1Test().run_tests()

