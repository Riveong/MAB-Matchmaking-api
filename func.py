import pandas as pd
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression


def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['age', 'gender'], axis=1)
    df = df.replace(0, pd.NA).dropna()
    scaler = StandardScaler().fit(df)
    df_scaled = scaler.transform(df)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_scaled)
    principalDf = pd.DataFrame(data=principalComponents, columns=['x', 'y'])
    return principalDf

def calculate_similarity(df):
    first_row = df.iloc[0:1]
    similarity_matrix = cosine_similarity(first_row, df)
    similarity_df = pd.DataFrame(similarity_matrix, columns=df.index)
    similarity_df['user_id'] = df.index[0]
    melted_df = similarity_df.melt(id_vars='user_id', var_name='match_id', value_name='similarity_score')
    threshold = 0.5
    melted_df['response'] = (melted_df['similarity_score'] >= threshold).astype(int)
    melted_df.loc[melted_df['match_id'] == 0, 'response'] = 0
    return melted_df

def epsilon_greedy_matchmaking(user_id, user_data, num_arms, epsilon, num_rounds):
    estimated_rewards = [0] * num_arms
    num_selected = [0] * num_arms
    total_reward = 0
    selected_arms = []

    for _ in range(num_rounds):
        if random.random() < epsilon:
            chosen_arm = random.choice(user_data['match_id'].unique())
        else:
            chosen_arm = max(range(num_arms), key=lambda arm: estimated_rewards[arm])

        reward_row = user_data[(user_data['user_id'] == user_id) & (user_data['match_id'] == chosen_arm)].sample()
        reward = reward_row['response'].values[0]
        total_reward += reward
        arm_index = list(user_data['match_id'].unique()).index(chosen_arm)
        num_selected[arm_index] += 1
        estimated_rewards[arm_index] += (reward - estimated_rewards[arm_index]) / num_selected[arm_index]
        if reward == 1:
            selected_arms.append(chosen_arm)

    return selected_arms, total_reward

def thompson_sampling_matchmaking(user_id, user_data, num_arms, num_rounds):
    alpha = np.ones(num_arms)
    beta = np.ones(num_arms)
    total_reward = 0
    selected_arms = []

    for _ in range(num_rounds):
        chosen_arm_index = np.argmax([np.random.beta(a + 1, b + 1) for a, b in zip(alpha, beta)])
        chosen_arm = user_data['match_id'].unique()[chosen_arm_index]
        
        reward_row = user_data[(user_data['user_id'] == user_id) & (user_data['match_id'] == chosen_arm)].sample()
        reward = reward_row['response'].values[0]
        total_reward += reward
        if reward == 1:
            selected_arms.append(chosen_arm)
        
        alpha[chosen_arm_index] += reward
        beta[chosen_arm_index] += (1 - reward)

    return selected_arms, total_reward

def linear_greedy_epsilon_matchmaking(user_id, user_data, num_arms, epsilon, num_rounds):
    model = LinearRegression()
    total_reward = 0
    selected_arms = []
    
    # Initialize the dataset for training the model
    training_data = pd.DataFrame()
    for arm in range(num_arms):
        arm_data = user_data[user_data['match_id'] == arm]
        if not arm_data.empty:
            training_data = pd.concat([training_data, arm_data.iloc[:1]], ignore_index=True)
    
    # Check if there are any entries in the training data before fitting
    if not training_data.empty:
        model.fit(training_data[['similarity_score']], training_data['response'])

        for _ in range(num_rounds):
            if random.random() < epsilon:
                # Explore
                chosen_arm = random.choice(user_data['match_id'].unique())
            else:
                # Exploit: Choose the arm with the highest predicted reward
                predictions = model.predict(user_data[['similarity_score']])
                user_data.loc[:, 'predicted_reward'] = predictions
                chosen_arm = user_data.loc[user_data['predicted_reward'].idxmax(), 'match_id']
            
            # Observe the reward and update the model accordingly
            reward_row = user_data[(user_data['user_id'] == user_id) & (user_data['match_id'] == chosen_arm)].sample()
            reward = reward_row['response'].values[0]
            total_reward += reward

            # If the reward is 1, add the arm to the list of selected arms
            if reward == 1:
                selected_arms.append(chosen_arm)
            
            # Add this data point to the training dataset and retrain the model
            if chosen_arm in training_data['match_id'].values:
                # Update the existing data point
                arm_index = training_data.index[training_data['match_id'] == chosen_arm].tolist()[0]
                training_data.loc[arm_index, 'response'] = reward
            else:
                # Add new data point
                training_data = pd.concat([training_data, reward_row], ignore_index=True)
            model.fit(training_data[['similarity_score']], training_data['response'])

    return selected_arms, total_reward

def master_function(filepath):
    # Preprocess data
    df = preprocess_data(filepath)
    
    # Calculate similarity
    preprocessed_data = calculate_similarity(df)
    
    # Save preprocessed data to a new CSV
    preprocessed_data.to_csv('preprocessed_data.csv')
    
    # Read user data
    user_id = 0
    user_data = preprocessed_data[preprocessed_data['user_id'] == user_id]
    num_arms = user_data['match_id'].nunique()
    num_rounds = 100
    epsilon = 0.5
    
    # Matchmaking tests
    selected_epsilon, epsilon_total_reward = epsilon_greedy_matchmaking(user_id, user_data, num_arms, epsilon, num_rounds)
    values, counts = np.unique(selected_epsilon, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_values = values[sorted_indices]
    print("Epsilon-Greedy Selected Arms:", sorted_values.tolist())
    print("Epsilon-Greedy Total Reward:", epsilon_total_reward)
    values, counts = np.unique(selected_epsilon, return_counts=True)
    sorted_indices = np.argsort(-counts)
    sorted_values = values[sorted_indices]
    print("Matchmaking Result:",sorted_values)
    
    selected_thompson, thompson_total_reward = thompson_sampling_matchmaking(user_id, user_data, num_arms, num_rounds)
    print("Thompson Sampling Selected Arms:", selected_thompson)
    print("Thompson Sampling Total Reward:", thompson_total_reward)

    selected_linear, linear_total_reward = linear_greedy_epsilon_matchmaking(user_id, user_data, num_arms, epsilon, num_rounds)
    print("lingreed Selected Arms:", selected_linear)
    print("lingreed Total Reward:", linear_total_reward)


    return[
        {
            "epsilon_reward":int(epsilon_total_reward),
            "epsilon_greedy":selected_epsilon,
        },
        {
            "thompson_reward":int(thompson_total_reward),
            "thompson_sampling":selected_thompson
        },
        {
            "lin_greedy_reward":int(linear_total_reward),
            "lin_greedy":selected_linear
        }
    ]
        



   
    

        
       