import gui_interaction.config
import tensor as ten

assert 0.5 == ten.normalise_point(0.5, 5)


assert 0.6 == ten.normalise_point(0.5, 5)

assert 0.7 == ten.normalise_point(0.5, 5)