from enterprise_ai_knowledge_assistant.services.employee_service import EmployeeService


def test_get_leave_balance_for_existing_employee() -> None:
    service = EmployeeService()

    result = service.get_employee_leave_balance(101)

    assert result == {"employee_id": 101, "name": "Palraj", "leave_balance": 8}


def test_get_leave_balance_for_another_employee() -> None:
    service = EmployeeService()

    result = service.get_employee_leave_balance(102)

    assert result == {"employee_id": 102, "name": "John", "leave_balance": 12}


def test_get_leave_balance_for_missing_employee_returns_error() -> None:
    service = EmployeeService()

    result = service.get_employee_leave_balance(999)

    assert result == {"error": "Employee with id 999 not found"}
