class EmployeeService:
    """Business logic for employees, using in-memory data for now."""

    def __init__(self) -> None:
        self._employees: dict[int, dict[str, str | int]] = {
            101: {"name": "Palraj", "leave_balance": 8},
            102: {"name": "John", "leave_balance": 12},
        }

    def get_employee_leave_balance(self, employee_id: int) -> dict[str, str | int]:
        """Return the employee's name and remaining leave days.

        A missing employee is returned as {"error": ...} rather than raised,
        because this result is sent back to the LLM, which can then explain
        the problem to the user in plain language.
        """
        employee = self._employees.get(employee_id)
        if employee is None:
            return {"error": f"Employee with id {employee_id} not found"}
        return {
            "employee_id": employee_id,
            "name": employee["name"],
            "leave_balance": employee["leave_balance"],
        }


# One shared instance, like document_service.
employee_service = EmployeeService()
