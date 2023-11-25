import tensorflow as tf
import numpy as np

# Define the Blackjack environment

class BlackjackEnvironment:
    # All card properties must be represented in a numerical value so the network understands. Think matrices... 
    class Card:
        def __init__(self, card_id, value, face_value, hand, suit):
            self.id = card_id
            self.played = 0  # 0 = false
            self.value = value  # actual numerical value of the card (face cards have 11)
            self.face_value = face_value  # -1 = jack -2 = queen -3 = king -4 = ace 
            self.hand = 0  #  0 for none, 1 for Kerry 2 for dealer
            self.suit = suit  # 0 = hearts 1 = diamonds 2 = clubs 3 = spades

    deck = []
    deck_state = [[]]

    def __init__(self):
        # Initially begin the reset process
        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        self.deck = self.generate_deck()

        # Draw two cards for the dealer
        dealer_card_1 = self.draw_card(self.deck)
        dealer_card_2 = self.draw_card(self.deck)

        if dealer_card_1 is not None and dealer_card_2 is not None:
            dealer_card_1.hand = 2
            dealer_card_2.hand = 2

        # Draw two cards for the player (network)
        player_card_1 = self.draw_card(self.deck)
        player_card_2 = self.draw_card(self.deck)

        if player_card_1 is not None and player_card_2 is not None:
            player_card_1.hand = 1
            player_card_2.hand = 1

        # Update the deck state
        self.deck_state = self.deck_to_2d_array(self.deck)

        # Return the updated state
        return self.deck_state

    # Generate the initial deck
    def generate_deck(self):
        deck = []
        card_id = 0
        for suit in range(4):
            for value in range(2, 11):
                card = self.Card(card_id, value, 0, 0, suit)
                deck.append(card)
                card_id += 1
            for face_value in range(-1, -5, -1):
                card = self.Card(card_id, 10, face_value, 0, suit)
                deck.append(card)
                card_id += 1
        return deck
    
    # This is refitting the deck to comply with the network state
    def deck_to_2d_array(self, deck):
        array = np.zeros((52, 5), dtype=int)
        for i, card in enumerate(deck):
            array[i] = [card.id, card.played, card.value, card.face_value, card.suit]
        return array

    def draw_card(self, deck):
        undealt_cards = [card for card in deck if not card.played]
        if not undealt_cards:
            # TODO: End episode here
            print("No more cards to draw.")
            return None
        
        drawn_card = np.random.choice(undealt_cards)
        drawn_card.played = 1
        return drawn_card

def step(self, action):
    # Action 0: Stand, Action 1: Hit
    if action == 0:
        # Player chooses to stand, let the dealer play
        while self.get_hand_value(self.deck_state[:, 2]) < 17:  # Dealer hits until reaching 17
            dealer_card = self.draw_card(self.deck)
            if dealer_card is not None:
                dealer_card.hand = 2
                self.deck_state = self.deck_to_2d_array(self.deck)
        
        # Determine the winner
        player_hand_value = self.get_hand_value(self.deck_state[:, 2])
        dealer_hand_value = self.get_hand_value(self.deck_state[:, 2], dealer=True)

        if player_hand_value > 21 or (dealer_hand_value <= 21 and dealer_hand_value >= player_hand_value):
            reward = -1  # Player loses
        elif dealer_hand_value > 21 or player_hand_value > dealer_hand_value:
            reward = 1  # Player wins
        else:
            reward = 0  # It's a draw

        done = True  # End the episode

    elif action == 1:
        # Player chooses to hit
        player_card = self.draw_card(self.deck)
        if player_card is not None:
            player_card.hand = 1
            self.deck_state = self.deck_to_2d_array(self.deck)

            # Check if player busts
            player_hand_value = self.get_hand_value(self.deck_state[:, 2])
            if player_hand_value > 21:
                reward = -1
                done = True  # End the episode
            else:
                reward = 0  # Continue the episode
                done = False

    return self.deck_state, reward, done

# Hyperparameters
state_size = 52 * 5  # Corrected state size based on the 2D array representation of card objects
action_size = 2  # 0 for 'stand', 1 for 'hit'
learning_rate = 0.001
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 32
num_episodes = 10000

# Our Network
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)

# Initialize environment network
env = BlackjackEnvironment()
model = QNetwork(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)
huber_loss = tf.keras.losses.Huber()

# Training loop
epsilon = epsilon_initial

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    with tf.GradientTape() as tape:
        for time in range(1000): 
            # Epsilon-greedy policy for action selection
            if np.random.rand() <= epsilon:
                action = np.random.choice(action_size)
            else:
                flat_state = state.flatten()  # Flatten the 2D array before feeding it to the model
                q_values = model(np.reshape(flat_state, [1, state_size]), training=True)
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            flat_state = state.flatten()  # Flatten the 2D array for the current state
            next_state_with_deck = np.concatenate([next_state, np.reshape(env.deck_state, [1, state_size])], axis=1)

            total_reward += reward

            target = reward + discount_factor * np.max(model(next_state_with_deck, training=True)[0])
            with tape.stop_recording():
                q_values = model(np.reshape(flat_state, [1, state_size]), training=True)
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
