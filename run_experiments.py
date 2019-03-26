import mlflow


def main():
    mlflow.run(
        uri='file:///C:\\Users\\mevsb\\PycharmProjects\\average_reward_cv',
        entry_point='main',
        parameters={
            'max_episodes': 1000,
            'environment': 'gridworld',
            'algorithm': 'n-step',
            'alpha': 0.1,
            'beta': 0.1,
            'n': 1,
            'lambda': 0,
            'seed': 1,
            'full_rbar': ''
        }
    )


if __name__ == '__main__':
    main()
