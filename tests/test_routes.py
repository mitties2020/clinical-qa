import unittest


class RouteRegistrationTests(unittest.TestCase):
    def test_app_import_and_unique_endpoints(self):
        from app import app

        endpoints = [rule.endpoint for rule in app.url_map.iter_rules()]
        # Flask reserves static endpoint; ensure all app-defined endpoints are unique.
        self.assertEqual(len(endpoints), len(set(endpoints)))

    def test_core_pages_registered_once(self):
        from app import app

        rules = list(app.url_map.iter_rules())
        by_path = {}
        for rule in rules:
            by_path.setdefault(rule.rule, []).append(rule)

        for path in ["/consultation-notes", "/dashboard", "/history", "/login"]:
            self.assertIn(path, by_path)
            self.assertEqual(len(by_path[path]), 1, f"{path} should be registered once")


if __name__ == "__main__":
    unittest.main()
