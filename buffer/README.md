The code is borrowed from [OpenAI/baselines](https://github.com/openai/baselines).

This implements the prioritized buffer to efficient sample rays whose reconstruction error is large.
The data in the buffer is no longer `(obs_t, action, reward, obs_tp1, done)`