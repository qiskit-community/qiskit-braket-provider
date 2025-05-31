"""Angle restrictions for different quantum backends."""

from math import pi
from typing import Optional, Union, Any
from braket.device_schema import DeviceCapabilities
from braket.device_schema.rigetti import RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2
from braket.device_schema.ionq import IonqDeviceCapabilities


def get_angle_restrictions(
    backend_properties: DeviceCapabilities, gate_name: str
) -> Optional[dict[str, Any]]:
    """Get angle restrictions for a specific gate on a backend.
    
    Args:
        backend_properties: Device capabilities/properties
        gate_name: Name of the gate (e.g., 'rx', 'ry', 'rz')
        
    Returns:
        Dictionary with restriction info or None if no restrictions
        Format: {'type': 'fixed_list', 'values': [pi, -pi, pi/2, -pi/2]}
               or {'type': 'range', 'min': -pi, 'max': pi}
    """
    if isinstance(backend_properties, (RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2)):
        return _get_rigetti_restrictions(gate_name)
    elif isinstance(backend_properties, IonqDeviceCapabilities):
        return _get_ionq_restrictions(gate_name)
    
    return None


def _get_rigetti_restrictions(gate_name: str) -> Optional[dict[str, Any]]:
    """Get Rigetti-specific angle restrictions."""
    if gate_name.lower() == 'rx':
        return {
            'type': 'fixed_list',
            'values': [pi, -pi, pi/2, -pi/2]
        }
    return None


def _get_ionq_restrictions(gate_name: str) -> Optional[dict[str, Any]]:
    """Get IonQ-specific angle restrictions."""
    if gate_name.lower() == 'ms':
        # MS gate has arbitrary angle range restrictions
        return {
            'type': 'range',
            'min': -pi,
            'max': pi
        }
    return None


def is_angle_allowed(angle: float, restrictions: dict[str, Any]) -> bool:
    """Check if an angle is allowed given restrictions.
    
    Args:
        angle: The angle to check
        restrictions: Restriction dictionary from get_angle_restrictions
        
    Returns:
        True if angle is allowed, False otherwise
    """
    if restrictions['type'] == 'fixed_list':
        # Check if angle is close to any allowed value (within tolerance)
        tolerance = 1e-10
        return any(abs(angle - allowed) < tolerance for allowed in restrictions['values'])
    elif restrictions['type'] == 'range':
        return restrictions['min'] <= angle <= restrictions['max']
    
    return True


def decompose_rx_rigetti(angle: float) -> list[tuple[str, list[float]]]:
    """Decompose RX(angle) into sequence of allowed Rigetti gates.
    
    For arbitrary RX(theta), we use: RZ(-pi/2) RX(pi/2) RZ(theta) RX(-pi/2) RZ(pi/2)
    
    Args:
        angle: The RX angle to decompose
        
    Returns:
        List of (gate_name, [angles]) tuples representing the decomposition
    """
    return [
        ('rz', [-pi/2]),
        ('rx', [pi/2]),
        ('rz', [angle]),
        ('rx', [-pi/2]),
        ('rz', [pi/2])
    ]