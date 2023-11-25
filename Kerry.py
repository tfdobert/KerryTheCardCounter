import tensorflow as tf
import numpy as np

# Define the Blackjack environment

class BlackjackEnvironment:
    class Card:
        def __init__(self, card_id, value, face_value, hand, suit):
            self.id = card_id
            self.played = False
            self.value = value
            self.face_value = face_value #such as suit or "jack". 
            self.hand = hand # whose hand the card is in, 0 for none, 1 for Kerry 2 for dealer
            self.hidden = False # We change this to true if it's a facedown card that Kerry cannot see
            self.suit = suit # This is completely optional


    # Creating a deck of cards
    suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    face_cards = ['Jack', 'Queen', 'King','Ace']
    deck = []
    def __init__(self):
        # Create the deck
        self.createDeck()


    def createDeck(self):
        print("Creating Fresh Deck...")
        self.deck = []
        card_id = 1  # Starting ID for cards
        for suit in self.suits:
            for rank in range(2, 11):  # Numerical cards 2-10
                card = self.Card(card_id, rank, str(rank), 0, suit)
                self.deck.append(card)
                card_id += 1

            for face_card in self.face_cards:
                card = self.Card(card_id, 10, face_card, 0, suit)
                self.deck.append(card)
                card_id += 1
        pass
    
    def deal_card(self,hand):
        # Deal a card from the deck
        # Get the indices of the remaining cards that have not been played
        remaining_indices = [i for i, (_, _, played) in enumerate(self.deck) if not played]

        if not remaining_indices:
            # If no remaining cards, initate episode end
            self.reset()

        # Randomly select an index from the remaining cards
        selected_index = np.random.choice(remaining_indices)
        
        # Mark the selected card as played and which hand it is in
        self.deck[selected_index] = (*self.deck[selected_index][:2], True)
        self.deck[selected_index].hand = hand

        # Return the entire selected card object
        return self.deck[selected_index]


    def reset(self):
        # Reset the environment to the initial state
        self.createDeck()
        self.players_hand = [0,0]
        self.dealers_hand = [0,0]

        # Reshape hands for the neural network input
        player_state = np.reshape(self.players_hand, [1, state_size])
        dealer_state = np.reshape([self.dealers_hand[0]], [1, state_size])

        # Reshape the deck state (assuming the deck is a list of 52 card objects)
        deck_state = np.reshape(self.deck, [1, deck_size])

        # Concatenate player's hand, dealer's face-up card, and the deck state
        state = np.concatenate([player_state, dealer_state, deck_state], axis=1)

        return state

    def step(self, action):
        # Initial turn parameters
        dealers_hand = []
        players_hand = []
        players_total = 0
        dealers_total = 0

        #TODO: hidden mechanic

        # Deal two cards for each player
        dealers_hand.append(self.deal_card(hand = 2))        
        dealers_hand.append(self.deal_card(hand = 2))

        players_hand.append(self.deal_card(hand = 1))
        players_hand.append(self.deal_card(hand = 1))

        # Now feed the state to the network
        q_values = model(state, training=True)

        action = 1

        #TODO: add a loop to allow the network to hit as many times as it likes.

        # Epsilon-greedy policy for action selection
        if np.random.rand() <= epsilon:
            action = np.random.choice(action_size)
        else:
            action = np.argmax(q_values[0])

        # Take the chosen action
        if action == 1:
            player_card = self.deal_card()
            self.players_hand.append(player_card)

            # Reshape the hands for the next state after hitting
            next_player_state = np.reshape(self.players_hand, [1, state_size])
            next_dealer_state = np.reshape(self.dealers_hand, [1, state_size])  # Updated to include the entire dealer's hand

            # Reshape the deck state for the next state
            next_deck_state = np.reshape(self.deck, [1, 52])

            # Concatenate player's hand, dealer's face-up card, and the deck state for the next state
            next_state = np.concatenate([next_player_state, next_dealer_state, next_deck_state], axis=1)

            if self.calculate_hand_value(self.players_hand) > 21:
                reward = -1
                done = True
            else:
                reward = 0
                done = False
        else:
            done = True
            while self.calculate_hand_value(self.dealers_hand) < 17:
                dealer_card = self.deal_card()
                self.dealers_hand.append(dealer_card)

            # Reshape the hands for the next state after the dealer hits
            next_player_state = np.reshape(self.players_hand, [1, state_size])
            next_dealer_state = np.reshape(self.dealers_hand, [1, state_size])

            # Reshape the deck state for the next state
            next_deck_state = np.reshape(self.deck, [1, 52])

            # Concatenate player's hand, dealer's face-up card, and the deck state for the next state
            next_state = np.concatenate([next_player_state, next_dealer_state, next_deck_state], axis=1)

            reward = self.determine_winner()

        return next_state, reward, done

# Hyperparameters
state_size = 52  # There will only ever be 52 cards
action_size = 2  # 0 for 'stand', 1 for 'hit'
deck_size = 52
learning_rate = 0.001
discount_factor = 0.99
epsilon_initial = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
batch_size = 32
num_episodes = 10000

# Our Network
class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, deck_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(state_size + deck_size,))
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)

# Initialize environment network
env = BlackjackEnvironment()
model = QNetwork(state_size, action_size, deck_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)
huber_loss = tf.keras.losses.Huber()

# Training loop
epsilon = epsilon_initial

for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    with tf.GradientTape() as tape:
        for time in range(1000): 
            # Epsilon-greedy policy for action selection
            if np.random.rand() <= epsilon:
                action = np.random.choice(action_size)
            else:
                # Concatenate player's hand, dealer's face-up card, and the deck state
                state_with_deck = np.concatenate([state, np.reshape(env.deck, [1, 52])], axis=1)
                q_values = model(state_with_deck, training=True)
                action = np.argmax(q_values[0])

            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Concatenate player's hand, dealer's face-up card, and the deck state for the next state
            next_state_with_deck = np.concatenate([next_state, np.reshape(env.deck, [1, 52])], axis=1)

            total_reward += reward

            target = reward + discount_factor * np.max(model(next_state_with_deck, training=True)[0])
            with tape.stop_recording():
                q_values = model(state_with_deck, training=True)
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
