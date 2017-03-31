from hypothesis import settings

settings.register_profile(
    'coverage', settings(max_examples=0),
)
