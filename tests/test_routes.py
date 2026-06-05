import unittest
from unittest.mock import patch


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


class FakeTwilioResponse:
    def __init__(self, status_code=201, body=None):
        self.status_code = status_code
        self._body = body or {"sid": "CA_fake"}
        self.headers = {"content-type": "application/json"}
        self.text = str(self._body)

    def json(self):
        return self._body


class TwilioCallingTests(unittest.TestCase):
    def authenticated_client(self):
        import app as app_module

        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        return app_module, client

    def test_call_patient_uses_app_base_url_and_returns_patient_sid(self):
        app_module, client = self.authenticated_client()
        env = {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_NUMBER": "+61411111111",
            "DOCTOR_PHONE": "0412222222",
            "APP_BASE_URL": "https://www.vividmedi.com",
            "BASE_URL": "",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(app_module.http, "post") as post:
            post.side_effect = [
                FakeTwilioResponse(body={"sid": "CA_doctor"}),
                FakeTwilioResponse(body={"sid": "CA_patient"}),
            ]
            response = client.post("/api/call-patient", json={"patientPhone": "0400 000 000"})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["patientSid"], "CA_patient")
        self.assertEqual(payload["sid"], "CA_patient")
        self.assertEqual(payload["doctorSid"], "CA_doctor")
        self.assertEqual(post.call_count, 2)
        doctor_call = post.call_args_list[0].kwargs["data"]
        patient_call = post.call_args_list[1].kwargs["data"]
        self.assertEqual(doctor_call["To"], "+61412222222")
        self.assertEqual(patient_call["To"], "+61400000000")
        self.assertTrue(doctor_call["Url"].startswith("https://www.vividmedi.com/twiml/join-consult"))
        self.assertEqual(patient_call["Method"], "POST")

    def test_call_patient_requires_doctor_phone_for_bridge_call(self):
        app_module, client = self.authenticated_client()
        env = {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_NUMBER": "+61411111111",
            "DOCTOR_PHONE": "",
            "ALLOW_PATIENT_ONLY_CALLS": "",
            "APP_BASE_URL": "https://www.vividmedi.com",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(app_module.http, "post") as post:
            response = client.post("/api/call-patient", json={"patientPhone": "0400 000 000"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("DOCTOR_PHONE", response.get_json()["error"])
        post.assert_not_called()

    def test_call_patient_rejects_local_callback_url(self):
        app_module, client = self.authenticated_client()
        env = {
            "TWILIO_ACCOUNT_SID": "AC123",
            "TWILIO_AUTH_TOKEN": "token",
            "TWILIO_NUMBER": "+61411111111",
            "DOCTOR_PHONE": "0412222222",
            "APP_BASE_URL": "http://127.0.0.1:5177",
            "BASE_URL": "",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(app_module.http, "post") as post:
            response = client.post("/api/call-patient", json={"patientPhone": "0400 000 000"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("public BASE_URL or APP_BASE_URL", response.get_json()["error"])
        post.assert_not_called()

    def test_join_consult_twiml_starts_media_stream_with_role_parameters(self):
        _app_module, client = self.authenticated_client()
        with patch.dict("os.environ", {"APP_BASE_URL": "https://www.vividmedi.com", "STREAM_URL": "", "TWILIO_STREAM_SECRET": ""}, clear=False):
            response = client.post("/twiml/join-consult?room=consult-test&role=patient")

        self.assertEqual(response.status_code, 200)
        xml = response.get_data(as_text=True)
        self.assertIn('<Start><Stream url="wss://www.vividmedi.com/twilio-stream" track="inbound_track">', xml)
        self.assertIn('<Parameter name="room" value="consult-test" />', xml)
        self.assertIn('<Parameter name="role" value="patient" />', xml)
        self.assertIn("<Dial><Conference", xml)


if __name__ == "__main__":
    unittest.main()
