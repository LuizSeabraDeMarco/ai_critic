from .critic import AICritic

def evaluate(model, X, y, **kwargs):
    critic = AICritic(**kwargs)
    return critic.evaluate(model, X, y)