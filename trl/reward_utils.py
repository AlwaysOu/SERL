def rewards_judge(prompts_p2, completions_p2, policy_rewards):
    rewards = []
    num_right = 0
    num_wrong = 0
    for j in range(len(completions_p2)):
        if completions_p2[j] == "1" or completions_p2[j] == "2":
            if completions_p2[j] == "1":
                predict_win_idx = prompts_p2[j]["index"][0]
                predict_loss_idx = prompts_p2[j]["index"][1]
            if completions_p2[j] == "2":
                predict_win_idx = prompts_p2[j]["index"][1]
                predict_loss_idx = prompts_p2[j]["index"][0]
            if policy_rewards[predict_win_idx] != policy_rewards[predict_loss_idx]:
                if (policy_rewards[predict_win_idx] - policy_rewards[predict_loss_idx]) > 0:
                    reward = 1
                    num_right += 1
                else:
                    reward = -1
                    num_wrong += 1
            else:
                reward = 0
        else:
            reward = -1
        rewards.append(reward)
    length = len(rewards)
    right = num_right/length
    wrong = num_wrong/length
    return rewards, right, wrong

def rewards_actor(len_completions, num_compares_onecompletion, count_completions, prompts_p2, completions_p2):
    rewards = []
    for i in range(len_completions):
        wins = 0
        for j in range(len(prompts_p2)):
            if i not in prompts_p2[j]["index"]:
                continue

            completion = completions_p2[j]
            if completion == '1':
                ans_idx = prompts_p2[j]["index"][0]
                other_idx = prompts_p2[j]["index"][1]
            elif completion == '2':
                ans_idx = prompts_p2[j]["index"][1]
                other_idx = prompts_p2[j]["index"][0]
            else:
                continue

            if i == ans_idx and count_completions[other_idx] * 1.2 > count_completions[ans_idx] and count_completions[other_idx]*0.8 < count_completions[ans_idx]:
                wins += count_completions[other_idx]/count_completions[ans_idx]
        reward = wins / num_compares_onecompletion
        rewards.append(reward)
    return rewards