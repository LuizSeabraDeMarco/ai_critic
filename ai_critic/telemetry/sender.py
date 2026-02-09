def send(payload: dict):
    try:
        requests.post(
            "https://api.ai-critic.dev/telemetry",
            json=payload,
            timeout=1
        )
    except Exception:
        pass  # nunca quebra o usuário
