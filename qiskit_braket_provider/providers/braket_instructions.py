
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate



""" IQM Experimental capabilities."""

class MeasureFF(Instruction):
    def __init__(self, feedback_key: int):
        """Measurement for Feed Forward control.

        Performs a measurement and stores the result in a classical feedback register
        for later use in conditional operations.
        
        Args:
            feedback_key (int): The integer feedback key that points to a measurement result.
        """
        super().__init__("MeasureFF", 1, 0, params=[feedback_key])

    broadcast_arguments = Gate.broadcast_arguments

    def __eq__(self, other):
        return (
            isinstance(other, MeasureFF) and self.feedback_key == other.feedback_key
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(feedback_key={self.params[0]})"


class CCPRx(Instruction):
    def __init__(self, angle_1: float, angle_2: float, feedback_key: int):
        """Classically controlled Phased Rx gate.

        A rotation around the X-axis with a phase factor, where the rotation depends
        on the value of a classical feedback.

        Args:
            angle_1 (float): The first angle of the gate in radians or
                expression representation.
            angle_2 (float): The second angle of the gate in radians or
                expression representation.
            feedback_key (int): The integer feedback key that points to a measurement result.
        """
        super().__init__("CCPRx", 1, 0, params=[angle_1, angle_2, feedback_key])

    broadcast_arguments = Gate.broadcast_arguments

    def __eq__(self, other):
        return (
            isinstance(other, CCPRx) 
            and self.feedback_key == other.feedback_key
            and self.angle_1 == other.angle_1
            and self.angle_2 == other.angle_2
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.angle_1}, {self.angle_2}, feedback_key={self.params[0]})"
