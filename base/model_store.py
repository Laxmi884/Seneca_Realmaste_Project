
import os
import pickle
import re
import json


class ModelStore:

    def set_prefix(self, prefix):
        self.prefix = prefix

    def save_model(self, filename, data, accuracy=0, meta: dict = None):
        """Save the model."""
        raise NotImplementedError

    def load_model(self, filename) -> tuple:  # (data, acuracy, meta)
        """Load the model."""
        raise NotImplementedError

    def delete_unused(self):
        """Delete unused model files."""
        raise NotImplementedError
