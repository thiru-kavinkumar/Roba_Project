from modules import *

# Environment initialization:
# ---------------------------
env = Environment()


# Main :
# -------
while env.masterRunning:
    env.clock.tick(env.masterFPS)
    # Event handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.masterRunning = False

        # Keyboard
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                env.masterRunning = False

    if (pygame.key.get_pressed()):
        # Robots keyboard coontroller
        env.move()

    env.draw()
    pygame.display.update()
