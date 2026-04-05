# env/action.py

ACTIONS = {
    0: "increase_ac",       # +cooling → comfort up, cost up
    1: "decrease_ac",       # -cooling → save energy, complaints may rise
    2: "lights_off_empty",  # lights off in unoccupied rooms
    3: "lights_on",         # restore lights
    4: "defer_heavy_load",  # shift heavy appliances to off-peak
    5: "do_nothing",        # hold current state
}

# Strategy mapping used by Task 3
# 0-1 = conservative, 2-3 = balanced, 4-5 = aggressive
ACTION_STRATEGY = {
    0: "conservative",
    1: "conservative",
    2: "balanced",
    3: "balanced",
    4: "aggressive",
    5: "aggressive",
}

def get_action_name(action_id):
    return ACTIONS.get(action_id, "unknown")

def get_action_strategy(action_id):
    return ACTION_STRATEGY.get(action_id, "balanced")

def get_action_count():
    return len(ACTIONS)