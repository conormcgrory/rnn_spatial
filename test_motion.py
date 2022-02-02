"""Script for testing motion simulation"""


import matplotlib.pyplot as plt

from motion import RectangleBoundary


def test_rectangle_boundary_contains():

    rect = RectangleBoundary(0.0, 0.0, 2.0, 2.0)

    res_1 = rect.contains(0.0, 0.0)
    print(f'Response: {res_1} (Correct: True)')

    res_2 = rect.contains(0.0, 5.0)
    print(f'Response: {res_2} (Correct: False)')

    res_3 = rect.contains(10.0, -10.0)
    print(f'Response: {res_3} (Correct: False)')


def test_rectangle_boundary_plot():

    rect = RectangleBoundary(0.0, 0.0, 2.0, 2.0)
    rect.plot()
    plt.show()


def main():

    #test_rectangle_boundary_contains()
    test_rectangle_boundary_plot()


if __name__ == '__main__':
    main()
