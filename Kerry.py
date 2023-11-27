import numpy as np
import tensorflow as tf
# Define the Blackjack environment

class BlackjackEnvironment:
    # All card properties must be represented in a numerical value so the network understands. Think matrices... 
    class Card:
        def __init__(self, card_id, value, face_value, hand, suit):
            self.id = card_id
            self.played = 0  # 0 = false
            self.value = value  # actual numerical value of the card (face cards have 10)
            self.face_value = face_value  # -1 = jack -2 = queen -3 = king -4 = ace 
            self.hand = hand  #  0 for none, 1 for Kerry 2 for dealer
            self.suit = suit  # 0 = hearts 1 = diamonds 2 = clubs 3 = spades

    deck = []
    deck_state = [[]]

    # Current hands
    dealers_hand = []
    players_hand = []


    def __init__(self):
        # Initially begin the reset process
        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        self.deck = self.generate_deck()

        # Initialize dealer and player parameters
        self.dealers_hand = []
        self.players_hand = []

        # Draw two cards for the dealer
        self.dealers_hand.append(self.draw_card(self.deck,2))
        self.dealers_hand.append(self.draw_card(self.deck,2))

        # Draw two cards for the player
        self.players_hand.append(self.draw_card(self.deck,1))
        self.players_hand.append(self.draw_card(self.deck,1))

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
        array = np.zeros((52, 6), dtype=int)
        for i, card in enumerate(deck):
            array[i] = [card.id, card.played, card.value, card.face_value, card.hand, card.suit]
        return array

    # Between games no one should have any cards in their hand. This function ensures that
    def reset_hands(self):
        for card in self.deck:
            card.hand = 0
        

    def draw_card(self, deck, player):
        undealt_cards = [card for card in deck if not card.played]
        if not undealt_cards:
            return None # No more cards to draw
        else:
            drawn_card = np.random.choice(undealt_cards)
            drawn_card.played = 1
            drawn_card.hand = player
            return drawn_card

    def step(self, action):
        done = False
        reward = 0

        # Calculate totals
        dealers_total = 0
        players_total = 0
        
        for dcard in self.dealers_hand:
            dealers_total += dcard.value
        for pcard in self.players_hand:
            players_total += pcard.value

        # Action 0: Stand, Action 1: Hit
        if action == 0:
            # Player chooses to stand, let the dealer play
            while dealers_total < 17:  # Dealer hits until reaching 17
                dealer_card = self.draw_card(self.deck, 2)
                if dealer_card == None:
                    break
                self.dealers_hand.append(dealer_card)
                dealers_total += dealer_card.value
                self.deck_state = self.deck_to_2d_array(self.deck)
            
            # Determine the winner
            if players_total > 21 or (dealers_total <= 21 and dealers_total >= players_total):
                reward = -1  # Player loses
                print("BOOOO!! Kerry loses, comeon Kerry you can do better!")
            elif dealers_total > 21 or players_total > dealers_total:
                reward = 1  # Player wins
                print("Thataboy Kerry!! Winner!!!")
            else:
                reward = 0  # It's a draw
                print("Stalemate, boring.")

            done = not any(card.played == 0 for card in self.deck)  # End the episode if no more cards can be drawn

        elif action == 1:
            # Player chooses to hit
            player_card = self.draw_card(self.deck, 1)
            if player_card != None:
                self.players_hand.append(player_card)
                players_total += player_card.value
                self.deck_state = self.deck_to_2d_array(self.deck)

                # Check if player busts
                if players_total > 21:
                    reward = -1
                    print("Oh no!! Kerry Busts! LOSER!!")
                    done = not any(card.played == 0 for card in self.deck)  # End the episode if no more cards can be drawn
                else:
                    reward = 0  # Continue the episode
                    done = False
            else:
                reward = 0  # No more cards to draw
                done = True
        
        self.dealers_hand = []
        self.players_hand = []
        self.reset_hands()
        return self.deck_state, reward, done
    

class BlackjackAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def select_action(self, state):
        # Use epsilon-greedy policy for exploration
        epsilon = 0.1
        if np.random.rand() <= epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state)
            legal_actions = np.arange(self.action_size)
            return np.random.choice(legal_actions[q_values[0, legal_actions] == np.max(q_values[0, legal_actions])])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            next_q_values = self.model.predict(next_state)
            target = reward + 0.95 * np.max(next_q_values)

        target_f = self.model.predict(state)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

# Assuming each card is represented by a 1D vector of length 6
state_size = 52 * 6
action_size = 2  # Stand and Hit

# Initialize the environment and agent
env = BlackjackEnvironment()
agent = BlackjackAgent(state_size, action_size)

# Training loop
episodes = 10000
for episode in range(episodes):
    state = env.reset()

    total_reward = 0
    done = False
    while not done:
        # Reshape the state to match the expected input shape
        state_flat = state.flatten()
        state_flat = np.reshape(state_flat, (1, state_size))

        action = agent.select_action(state_flat)
        next_state, reward, done = env.step(action)

        # Reshape the next_state for training
        next_state_flat = next_state.flatten()
        next_state_flat = np.reshape(next_state_flat, (1, state_size))

        agent.train(state_flat, action, reward, next_state_flat, done)
        total_reward += reward
        state = next_state

    if episode % 100 == 0:
        print("Episode {}: Total Reward: {}".format(episode, total_reward))
