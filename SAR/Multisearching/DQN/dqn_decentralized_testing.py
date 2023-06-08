
def test_dqn():
    for policy in policy_num:
        debug = True
        print("Testing policy %d:" %(policy))
        file_name = "hyperparameters%s.json" %(str(policy))
        ts, nr, ep, lr, dr, er, pr, ner, per, nsr, ms, n_actions, c_dims, k_size, s_size, fc_dims, prioritized, mem_size, batch_size, r, _ = read_hp_json(load_path, file_name)
        # int(ts),float(lr), float(dr), float(er), float(pr), float(nr), float(per), float(nsr), int(ms), int(n_actions), c_dims, k_size, s_size, fc_dims,int(mem_size), int(batch_size), int(replace),env_size
        
        agents = []
        for i in range(nr):
            agents.append(DQNAgent(nr, dr, er[0], er[1], er[2], lr,
                        n_actions, starting_beta, input_dims,
                        c_dims, k_size, s_size, fc_dims,
                        mem_size,
                        batch_size, r, prioritized,
                        algo='DQNAgent', env_name=env_size, chkpt_dir=models_path))
        for r_i, agent in enumerate(agents):
            file_name = "agent%s_experience%s_checkpoint.pth" %(str(r_i),str(policy))
            file_name = os.path.join(load_path, file_name)
            agent.q_eval.load_state_dict(T.load(file_name, map_location='cuda:0'))
            agent.q_eval.eval()
            agent.q_next.load_state_dict(T.load(file_name, map_location='cuda:0'))
            agent.q_next.eval()
        
        env = Environment(nr, pr, ner, per, nsr)

        PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])

        trajs = []
        steps = []
        cnt = [0]*nr
        trajectories = []
        fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
        for i in range(0, testing_iterations):
            if i % 1000 == 0 and i != 0:
                print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt[0]+cnt[1])/float(i)*100, np.mean(np.array(steps))))
            observation = env.reset()

            trajectory = []

            done = False
            

            # if show_plot:
            #     PR.print_trajectories(ax, save_path, policy, env)
        
            for step in range(int(ms)):
                if step == 0: actions = [None]
                if show_plot:
                    plt.cla()
                    PR.print_trajectories(ax, save_path, policy, env, actions[0])
                    # breakpoint
                actions = []
                action = [0]*nr
                for i_r in range(0,nr):
                    action[i_r] = agent.choose_action(observation[i_r])
                    trajectory.append((env.pos[i_r], action[i_r], i_r))
                actions.append(action)
                observation_, reward, done, info = env.step(action)
                if info != None:
                    for j in range(nr):
                        if info == j:
                            cnt[j] += 1

                observation = observation_

                if done:
                    if show_plot:
                        plt.cla()
                        PR.print_trajectories(ax, save_path, policy, env, actions[0])
                        if not info: plt.pause(1)
                    trajectories.append(trajectory)
                    break
                
            steps.append(step)
            # if step == int(ms)-1 and not done:
            #     trajectories.append(trajectory)

        p = (cnt[0]+cnt[1])/(testing_iterations)*100
        for i in range(nr):
            print("Percentage success for drone %d: %d / %d x 100 = %.2f %%" %(i, cnt[i], testing_iterations, (cnt[i])/(testing_iterations)*100))
        print("Total percentage success: %d / %d x 100 = %.2f %%" %(cnt[0]+cnt[1], testing_iterations, p))
        print("Average steps: %.2f" %(np.mean(np.array(steps))))
        file_name = "Results%s.json" %(str(policy))
        file_name = os.path.join(load_path, file_name)
        write_json("Success:%s, Average steps:%s" %(str(p), str(np.mean(np.array(steps)))), file_name)