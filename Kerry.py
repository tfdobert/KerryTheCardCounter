import tensorflow as tf
import numpy as np

# Define the Blackjack environment

class BlackjackEnvironment:
    class Card:
        def __init__(self, card_id, value, card_type):
            self.id = card_id
            self.played = False
            self.value = value
            self.card_type = card_type

    # Creating a deck of cards
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    face_cards = ['Jack', 'Queen', 'King','Ace']
    deck = []
    players_hand = []
    dealers_hand = []
    def __init__(self):
        # Create the deck
        self.createDeck()


    def createDeck(self):
        self.deck = []
        card_id = 1  # Starting ID for cards
        for suit in self.suits:
            for rank in range(2, 11):  # Numerical cards 2-10
                card = self.Card(card_id, rank, str(rank))
                self.deck.append(card)
                card_id += 1

            for face_card in self.face_cards:
                card = self.Card(card_id, 10, face_card)
                self.deck.append(card)
                card_id += 1
        pass
    
    def deal_card(self):
        # Deal a card from the deck
        # Get the indices of the remaining cards that have not been played
        remaining_indices = [i for i, (_, _, played) in enumerate(self.deck) if not played]

        if not remaining_indices:
            # If no remaining cards, initate episode end
            self.reset()

        # Randomly select an index from the remaining cards
        selected_index = np.random.choice(remaining_indices)
        # Mark the selected card as played
        self.deck[selected_index] = (*self.deck[selected_index][:2], True)

        # Return the entire selected card object
        return self.deck[selected_index]


    def reset(self):
            # Reset the environment to the initial state
            self.createDeck()
            pass

    def step(self, action):
        # Inital turn parameters
        players_total = 0
        dealers_total = 0
        
        # Action: 0 for 'stand', 1 for 'hit'
        if action == 1:
            # Agent chooses to hit, deal a card to agent
            player_card = self.deal_card()
            self.player_hand.append(player_card)

            # Check if the player busts
            if self.calculate_hand_value(self.player_hand) > 21:
                reward = -1  # Player busts, receives a negative reward
                done = True
            else:
                reward = 0  # No bust, continue the game
                done = False
        else:
            # If the agent chooses to stick, play out the dealer's turn
            done = True
            while self.calculate_hand_value(self.dealer_hand) < 17: # Dealers typically only hit on a 16 or lower
                dealer_card = self.deal_card()
                self.dealer_hand.append(dealer_card)

            # Determine the winner
            reward = self.determine_winner()

        # Return the new state (player's hand, dealer's face-up card), reward, and whether the episode is done
        return (self.player_hand, self.dealer_hand[0]), reward, done
        pass

# Define the Q-network architecture
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Define your neural network layers here

    def call(self, state):
        # Define the forward pass of your neural network
        pass

# Hyperparameters
state_size = 21  # Change this based on your state representation
action_size = 2  # 0 for 'stick', 1 for 'hit'
learning_rate = 0.001
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 32
num_episodes = 10000

# Initialize environment and Q-network
env = BlackjackEnvironment()
model = QNetwork(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)
huber_loss = tf.keras.losses.Huber()

# Training loop
epsilon = epsilon_initial

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    with tf.GradientTape() as tape:
        for time in range(1000):  # You may adjust the maximum number of time steps
            # Epsilon-greedy policy for action selection
            if np.random.rand() <= epsilon:
                action = np.random.choice(action_size)
            else:
                q_values = model(state, training=True)
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            total_reward += reward

            target = reward + discount_factor * np.max(model(next_state, training=True)[0])
            with tape.stop_recording():
                q_values = model(state, training=True)
                loss = huber_loss(target, q_values[0][action])

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            state = next_state

            if done:
                break

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Print results
    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# Save the trained model if needed
model.save("blackjack_model")
