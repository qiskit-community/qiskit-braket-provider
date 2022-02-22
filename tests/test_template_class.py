"""Tests for template."""
from unittest import TestCase

from prototype_template.template_class import TemplateClass


class TestPrototypeTemplate(TestCase):
    """Tests prototype template."""

    def test_template_class(self):
        """Tests template class."""
        obj = TemplateClass(1)

        self.assertEqual(obj.multiply(2), 2)
        self.assertEqual(repr(obj), "TemplateClass(1)")
