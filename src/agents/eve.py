class Eve:
    def __init__(self):
        self.suspicious_words = ['insider', 'confidential', 'secret', 'reliable source']

    def analyze_communication(self, message: str, thought: str):
        message_suspicious = any(word in message.lower() for word in self.suspicious_words)
        thought_suspicious = any(word in thought.lower() for word in self.suspicious_words)
        return message_suspicious or thought_suspicious

    def interject(self, message: str, thought: str):
        if self.analyze_communication(message, thought):
            return "Eve: Suspicious activity detected. Communication halted."
        return None