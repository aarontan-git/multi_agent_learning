# Pseudocode

reward_run_all = []
reward_epsilon = []
test_reward_epsilon = []
test_reward_run_all

for epsilon

    reward_run = []
    test_reward_run = []
    Q_values_list = []

    for run

        reward_episode = []
        test_reward_episode = []
        Q_values
        delta_list = []

        for eps 
        
            reward_list = []

            for steps

                calculate delta
                get_reward

                reward_list.append(reward)

            reward_episode.append(sum(reward_list))

            delta_list.append(delta)

            # TESTING
            get_test_reward
            test_reward_episode.append(sum(test_reward))

        Q_values_list.append(Q_values)
        test_reward_run.append(average(test_reward_list))
        reward_run.append(average(reward_episode))

        plot(reward_episode, test_reward_episode, delta_list)
    
    reward_run_all.append(reward_run)
    test_reward_run_all.append(test_reward_run)
    test_reward_epsilon.append(average(test_reward_run))
    reward_epsilon.append(average(reward_run))

plot(reward_epsilon, reward_run_all, test_reward_run_all, test_reward_epsilon)