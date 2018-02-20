from hypothesis import settings, HealthCheck

settings.register_profile(
    'default',
    settings(
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    ))

settings.load_profile('default')

settings.register_profile(
    'coverage', settings(max_examples=0),
)
