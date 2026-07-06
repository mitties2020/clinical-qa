import base64
import hashlib
import hmac
import unittest
from io import BytesIO
from unittest.mock import patch


def twilio_signature(url, auth_token, params=None):
    signed_data = url
    for key, value in sorted((params or {}).items()):
        signed_data += f"{key}{value}"
    digest = hmac.new(auth_token.encode("utf-8"), signed_data.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


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

    def test_wa_mental_health_discharge_summary_prompt_uses_psychiatry_structure(self):
        import app as app_module

        prompt = app_module.build_consult_prompt_context("WA mental health discharge summary")

        self.assertIn("WA hospital psychiatry discharge summary", prompt)
        self.assertIn("Problems this Admission", prompt)
        self.assertIn("Mental State on Admission", prompt)
        self.assertIn("Risk Assessment on Discharge", prompt)
        self.assertIn("Advice to Community Mental Health Team", prompt)
        self.assertIn("multiple pasted admission notes", prompt.lower())
        self.assertIn("do not force it to be short", prompt)
        self.assertIn("senior psychiatry registrar/consultant", prompt)
        self.assertIn("If the patient died during admission", prompt)
        self.assertIn("do not claim WA Health compliance is guaranteed", prompt)

    def test_wa_mental_health_discharge_summary_gets_long_completion_budget(self):
        import app as app_module

        self.assertGreaterEqual(app_module.consult_completion_budget("WA mental health discharge summary"), 6000)
        self.assertGreaterEqual(app_module.consult_request_timeout("WA mental health discharge summary"), 150)
        self.assertEqual(app_module.consult_completion_budget("General consultation note"), 1800)


class AuthenticationTests(unittest.TestCase):
    def test_authentication_fails_closed_without_auth_code(self):
        import app as app_module

        old_auth_code = app_module.AUTH_CODE
        app_module.AUTH_CODE = ""
        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
        try:
            response = client.post("/authenticate", json={})
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.get_json()["ok"])
            with client.session_transaction() as sess:
                self.assertIsNone(sess.get("authenticated"))
        finally:
            app_module.AUTH_CODE = old_auth_code

    def test_authentication_requires_non_empty_matching_code(self):
        import app as app_module

        old_auth_code = app_module.AUTH_CODE
        app_module.AUTH_CODE = "123456"
        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
        try:
            empty_response = client.post("/authenticate", json={})
            self.assertEqual(empty_response.status_code, 401)

            good_response = client.post("/authenticate", json={"code": "123456"})
            self.assertEqual(good_response.status_code, 200)
            self.assertTrue(good_response.get_json()["ok"])
            with client.session_transaction() as sess:
                self.assertIs(sess.get("authenticated"), True)
        finally:
            app_module.AUTH_CODE = old_auth_code


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
        with patch.dict("os.environ", {"APP_BASE_URL": "https://www.vividmedi.com", "STREAM_URL": "", "TWILIO_STREAM_SECRET": "", "TWILIO_AUTH_TOKEN": ""}, clear=False):
            response = client.post("/twiml/join-consult?room=consult-test&role=doctor")

        self.assertEqual(response.status_code, 200)
        xml = response.get_data(as_text=True)
        self.assertIn('<Start><Stream name="consult-test-doctor" url="wss://www.vividmedi.com/twilio-stream" track="both_tracks"', xml)
        self.assertIn('statusCallback="https://www.vividmedi.com/api/stream-status?room=consult-test&amp;role=doctor"', xml)
        self.assertIn('<Parameter name="room" value="consult-test" />', xml)
        self.assertIn('<Parameter name="role" value="doctor" />', xml)
        self.assertIn("<Dial><Conference", xml)

    def test_patient_leg_does_not_duplicate_stream_by_default(self):
        _app_module, client = self.authenticated_client()
        with patch.dict("os.environ", {"APP_BASE_URL": "https://www.vividmedi.com", "STREAM_URL": "", "TWILIO_STREAM_LEG": "", "TWILIO_AUTH_TOKEN": ""}, clear=False):
            response = client.post("/twiml/join-consult?room=consult-test&role=patient")

        self.assertEqual(response.status_code, 200)
        xml = response.get_data(as_text=True)
        self.assertNotIn("<Start><Stream", xml)
        self.assertIn("<Dial><Conference", xml)

    def test_doctor_stream_labels_tracks_as_clinician_and_patient(self):
        import app as app_module

        self.assertEqual(app_module.twilio_track_speaker_label("doctor", "inbound"), "Clinician")
        self.assertEqual(app_module.twilio_track_speaker_label("doctor", "outbound"), "Patient")

    def test_join_consult_rejects_unsigned_twilio_request_when_token_configured(self):
        _app_module, client = self.authenticated_client()
        env = {
            "APP_BASE_URL": "https://www.vividmedi.com",
            "STREAM_URL": "",
            "TWILIO_AUTH_TOKEN": "twilio-token",
            "TWILIO_STREAM_SECRET": "secret-value",
            "TWILIO_VALIDATE_SIGNATURE": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            response = client.post(
                "/twiml/join-consult?room=consult-test&role=doctor",
                base_url="https://www.vividmedi.com",
            )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn("secret-value", response.get_data(as_text=True))

    def test_join_consult_requires_auth_token_before_emitting_stream_secret(self):
        _app_module, client = self.authenticated_client()
        env = {
            "APP_BASE_URL": "https://www.vividmedi.com",
            "STREAM_URL": "",
            "TWILIO_AUTH_TOKEN": "",
            "TWILIO_STREAM_SECRET": "secret-value",
            "TWILIO_VALIDATE_SIGNATURE": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            response = client.post("/twiml/join-consult?room=consult-test&role=doctor")

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("secret-value", response.get_data(as_text=True))

    def test_signed_join_consult_can_emit_stream_secret_for_twilio(self):
        _app_module, client = self.authenticated_client()
        url = "https://www.vividmedi.com/twiml/join-consult?room=consult-test&role=doctor"
        env = {
            "APP_BASE_URL": "https://www.vividmedi.com",
            "STREAM_URL": "",
            "TWILIO_AUTH_TOKEN": "twilio-token",
            "TWILIO_STREAM_SECRET": "secret-value",
            "TWILIO_VALIDATE_SIGNATURE": "true",
        }
        with patch.dict("os.environ", env, clear=False):
            response = client.post(
                "/twiml/join-consult?room=consult-test&role=doctor",
                base_url="https://www.vividmedi.com",
                headers={"X-Twilio-Signature": twilio_signature(url, "twilio-token")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("secret-value", response.get_data(as_text=True))

    def test_call_status_records_metric_without_crashing(self):
        app_module, client = self.authenticated_client()
        with patch.dict("os.environ", {"TWILIO_AUTH_TOKEN": ""}, clear=False):
            response = client.post("/api/call-status")

        self.assertEqual(response.status_code, 204)
        self.assertIn("twilio.call_status.webhook", app_module.monitor.system_metrics)


class MicTranscriptionTests(unittest.TestCase):
    def authenticated_client(self):
        import app as app_module

        app_module.app.config.update(TESTING=True)
        client = app_module.app.test_client()
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        return app_module, client

    def test_transcribe_uses_deepgram_when_configured(self):
        app_module, client = self.authenticated_client()
        deepgram_body = {
            "results": {
                "channels": [
                    {"alternatives": [{"transcript": "hello this is a test"}]}
                ]
            }
        }
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "dg-token"}, clear=False), patch.object(app_module.http, "post") as post:
            post.return_value = FakeTwilioResponse(body=deepgram_body)
            response = client.post(
                "/api/transcribe",
                data={"audio": (BytesIO(b"fake-webm-audio"), "dictation.webm")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "hello this is a test")
        self.assertIn("https://api.deepgram.com/v1/listen", post.call_args.args[0])
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Token dg-token")

    def test_transcribe_rejects_oversized_upload(self):
        app_module, client = self.authenticated_client()
        old_limit = app_module.MAX_AUDIO_UPLOAD_BYTES
        old_config_limit = app_module.app.config.get("MAX_CONTENT_LENGTH")
        app_module.MAX_AUDIO_UPLOAD_BYTES = 4
        app_module.app.config["MAX_CONTENT_LENGTH"] = None
        try:
            response = client.post(
                "/api/transcribe",
                data={"audio": (BytesIO(b"too-large"), "dictation.webm")},
                content_type="multipart/form-data",
            )
        finally:
            app_module.MAX_AUDIO_UPLOAD_BYTES = old_limit
            app_module.app.config["MAX_CONTENT_LENGTH"] = old_config_limit

        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
