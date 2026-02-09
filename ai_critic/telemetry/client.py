def collect_and_send(report, enabled=True):
    if not enabled:
        return

    payload = anonymize(report)
    send(payload)
