from bank_customer_complaint_analysis import Calculator


def test_add():
    assert Calculator.add(2, 3) == 5


def test_subtract():
    assert Calculator.subtract(5, 3) == 2
