def linear_schedule(initial_value, min_value = 0.0):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        min_value = float(min_value)
        assert (initial_value > 0.0 ), "linear_schedule work only with positive decreasing values"

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return max(progress * initial_value, min_value)

    return func
