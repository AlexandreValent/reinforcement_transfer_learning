import gym

class Planet:
    """
    planet string
    human_render True or False

    gravity -12.0 < gravity and gravity < 0.0
    enable_wind True  False
    wind_power 0.0 > wind_power or wind_power > 20.0
    turbulence_power 0.0 > turbulence_power or turbulence_power > 2.0
    """
    def __init__(self, planet,human_render):
        self.name = planet
        self.human_render = human_render

        self.env = None # to know if env is load

        # Set render value
        if self.human_render:
            render_mode = "human"
        else :
            render_mode = "rgb_array"

        if self.name == "Earth":
            self.env = gym.make("LunarLander-v2",
                            render_mode = render_mode,
                            gravity=-9.8,
                            enable_wind=True,
                            wind_power = 5,
                            turbulence_power=0.5)

        if self.name == "Moon":
            self.env = gym.make("LunarLander-v2",
                            render_mode = render_mode,
                            gravity=-1.6,
                            enable_wind=False)

        if self.name == "Neptune":
            self.env = gym.make("LunarLander-v2",
                            render_mode = render_mode,
                            gravity=-11,
                            enable_wind=True,
                            wind_power = 20,
                            turbulence_power=2)

        if self.name == "Asteroid ":
            self.env = gym.make("LunarLander-v2",
                            render_mode = render_mode,
                            gravity=-0.2,
                            enable_wind=False)

        if self.env == None:
            print("❌ Unable to create an env. Please, select an existing plannet.")

        else :
            print("✅ Environment successfully created.")
