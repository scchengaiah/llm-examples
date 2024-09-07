from agency_swarm.tools import BaseTool
from pydantic import Field

class CalculatorTool(BaseTool):
    """
    Tool for performing mathematical calculations such as sum, minus, multiplication, division, etc.
    """
    operation: str = Field(..., description="A mathematical expression to be evaluated, e.g., '200*7' or '5000/2*10'.")

    def run(self):
        try:
            result = eval(self.operation)
            return str(result)
        except SyntaxError:
            return "Error: Invalid syntax in mathematical expression"
