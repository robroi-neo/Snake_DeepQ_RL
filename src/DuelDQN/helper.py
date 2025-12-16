import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def save_plot(scores, mean_scores, iteration):
    if iteration % 250 != 0:
        return  # do nothing except every 250th iteration

    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)

    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.savefig(f"plots/DuelDQN/plot_{iteration}.png")
    plt.close()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

def distance_to_body(head, body, w, h):
    # returns normalized smallest distance in each direction
    up = down = left = right = 1.0  # default no-body-in-direction

    for s in body[1:]:  # skip head
        if s.x == head.x and s.y < head.y:    # up
            up = min(up, (head.y - s.y) / h)
        if s.x == head.x and s.y > head.y:    # down
            down = min(down, (s.y - head.y) / h)
        if s.y == head.y and s.x < head.x:    # left
            left = min(left, (head.x - s.x) / w)
        if s.y == head.y and s.x > head.x:    # right
            right = min(right, (s.x - head.x) / w)

    return up, down, left, right
