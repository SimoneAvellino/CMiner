__all__ = ["CMiner"]


def __getattr__(name):
	if name == "CMiner":
		from .CMiner import CMiner

		return CMiner
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")