import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DeploymentDocsTests(unittest.TestCase):
    def test_deployment_notes_explain_requirements_shape(self):
        text = (ROOT / "DEPLOYMENT.md").read_text()

        self.assertIn("requirements.txt", text)
        self.assertIn("dependency import", text)
        self.assertIn("requirements-core.txt", text)


if __name__ == "__main__":
    unittest.main()
