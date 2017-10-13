import argparse
import random

import matplotlib.pyplot as plt

random.seed(42)


class Bird(object):
    def __init__(self, strategy, health):
        assert strategy in ['hawk', 'dove']
        self._strategy = strategy
        self._health = health


    @property
    def health(self):
        return self._health

    @health.setter
    def health(self, value):
        self._health = value

    @property
    def strategy(self):
        return self._strategy



def simulate(N, events, random_init=True, frac_doves=None, 
             win=50, lose=0, initial_health=0, time_penalty=10, 
             injury=100, verbose=True, plot=True):
    """
    Arguments:
        N            -- population size, remains fixed
        events       -- number of events (strategy showdowns) to simulate
        random_init  -- randomly initialize strategies within population?
        frac_doves   -- if not random_init, initial fraction of 'doves' in population
        win          -- the payoff for winning an event
        lose         -- the cost of losing an event
        time_penalty -- the cost to doves for engaging and losing
        injury       -- the cost to hawks for engaging and losing
        verbose      -- print out population counts
        plot         -- display and save populations as function of events
    
    Returns:
        hawk_counts    -- list of 100 hawk population counts
        dove_counts    -- list of 100 dove population counts
        fraction_hawks -- list of fraction of hawks in population

    """
    birds = ['hawk', 'dove']
    if random_init:
        population = [Bird(random.choice(birds), initial_health) for _ in range(N)]
    else:
        assert frac_doves is not None
        num_doves = int(frac_doves * N)
        population = [Bird('dove', initial_health) for _ in range(num_doves)] + \
                     [Bird('hawk', initial_health) for _ in range(N - num_doves)]

    hawk_counts = [len([p for p in population if p.strategy == 'hawk'])]
    dove_counts = [len([p for p in population if p.strategy == 'dove'])]
    fraction_hawks = [float(hawk_counts[-1]) / len(population)]

    for event in range(events):
        i = random.choice(range(len(population)))
        j = random.choice(range(len(population)))

        b1 = population[i]
        b2 = population[j]

        # 'loser' (i.e. i or j) can be arbitrarily chosen since
        # the selection is already random
        if b1.strategy == 'dove' and b2.strategy == 'dove':
            b1.health = b1.health - time_penalty
            b2.health = b2.health + (win - time_penalty)
        elif b1.strategy == 'dove' and b2.strategy == 'hawk':
            b1.health = b1.health - lose
            b2.health = b2.health + win
        elif b1.strategy == 'hawk' and b2.strategy == 'dove':
            b1.health = b1.health + win
            b2.health = b2.health - lose
        elif b1.strategy == 'hawk' and b2.strategy == 'hawk':
            b1.health = b1.health - injury
            b2.health = b2.health + win
        else:
            raise ValueError('Unkown strategy encountered (%s, %s)' (b1.strategy, b2.strategy))

        # check if either involved 'died', i.e. health
        # goes to 0 or below.  if so, replace with the 
        # type of a randomly selected 'bird'
        if b1.health < 0:
            k = random.choice([x for x in range(len(population)) if x != i])
            reproducer = population[k]
            population[i] = Bird(reproducer.strategy, initial_health)
        if b2.health < 0:
            k = random.choice([x for x in range(len(population)) if x != j])
            reproducer = population[k]
            population[j] = Bird(reproducer.strategy, initial_health)


        if event % (events // 100) == 0 and event > 0:
            hawk_counts.append(len([p for p in population if p.strategy == 'hawk']))
            dove_counts.append(len([p for p in population if p.strategy == 'dove']))
            fraction_hawks.append(float(hawk_counts[-1]) / len(population))

            if verbose and event % (events // 10) == 0:
                print('Events: %d' % event)
                print('Hawks: %d, Doves: %d' % (hawk_counts[-1], dove_counts[-1]))
                print('Fraction hawks: %s' % str(fraction_hawks[-1]))
                print('------------------------')


    def simulation_plot():
        plt.subplot(211)
        plt.plot(range(len(hawk_counts)), hawk_counts, label='Hawks')
        plt.plot(range(len(dove_counts)), dove_counts, label='Doves')
        plt.title('Dove and Hawk Populations Over Time')
        plt.ylabel('Population size')
        plt.legend(loc='best')
        plt.subplot(212)
        plt.plot(range(len(fraction_hawks)), fraction_hawks)
        plt.title('Population Ratio')
        plt.xlabel('Number of events (1000s)')
        plt.ylabel('Fraction of Hawk Population')
        plt.savefig('figs/hawks_doves_%d_%d.png' % (N, events))
        plt.show()

    if plot:
        simulation_plot()

    return hawk_counts, dove_counts, fraction_hawks



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate ESS strategies for hawks and doves.')
    parser.add_argument('--N', type=int, 
                        help='size of the overall population (fixed).')
    parser.add_argument('--events', type=int, 
                        help='number of events (fights with payoff) to simulate.')
    parser.add_argument('--win', type=int, 
                        help='payoff for winning an event.')
    parser.add_argument('--lose', type=int, 
                        help='cost of losing an event.')
    parser.add_argument('--health', type=int, 
                        help='initial health of population members.')
    parser.add_argument('--tpenalty', type=int, 
                        help='cost to dove to posture.')
    parser.add_argument('--injury', type=int, 
                        help='cost to hawk to lose fight.')

    args = parser.parse_args()
    hawks, doves, frachawks = simulate(N=args.N, 
                                       events=args.events, 
                                       win=args.win, 
                                       lose=args.lose, 
                                       initial_health=args.health,
                                       time_penalty=args.tpenalty,
                                       injury=args.injury)
