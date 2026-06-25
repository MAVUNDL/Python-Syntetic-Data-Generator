import numpy as np
from dataclasses import dataclass, field
from datetime import date, timedelta

""" This is the Base Distribution class """
class Distribution:
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        raise NotImplementedError
    

""" Continuous Distribution Subclasses """

@dataclass
class Uniform(Distribution):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return gen.uniform(self.min, self.max, n)
    

@dataclass
class Normal(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return gen.normal(self.mean, self.std, n)
    

@dataclass
class LogNormal(Distribution):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return gen.lognormal(self.mu, self.sigma, n)
    

@dataclass
class Exponential(Distribution):
    def __init__(self):
        self.scale: float = 1.0
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return gen.exponential(self.scale, n)


@dataclass
class Beta(Distribution):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return gen.beta(self.a, self.b, n)
    

@dataclass
class Gamma(Distribution):
    def __init__(self, shape: float):
        self.shape = shape
        self.scale: float = 1.0
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return gen.gamma(self.shape, self.scale, n)


@dataclass
class Weibull(Distribution):
    def __init__(self, a: float):
        self.a = a
        self.scale: float = 1.0

    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        return self.scale * gen.weibull(self.a, n)
    

@dataclass
class Mixture(Distribution):
    def __init__(self, components: list[Distribution], weights: list[float]):
        self.components = components
        self.weights = weights
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        weights = np.array(self.weights)
        weights = weights / weights.sum()
        counts = gen.multinomial(n, weights)
        parts = [c.sample(int(ctn), gen) for c, ctn in zip(self.components, counts)]
        out = np.concatenate(parts)
        gen.shuffle(out)
        return out

    
""" Categorical Distribution Subclasses"""

@dataclass
class WeightedChoice(Distribution):
    def __init__(self, values: list[any], weights: list[float]):
        self.values = values
        self.weigths = weights
    
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        w = np.array(self.weigths, dtype=float)
        w = w / w.sum()
        index = gen.choice(len(self.values), size=n, p=w)
        return np.array(self.values)[index]


@dataclass
class WeightedChoiceMapping(Distribution):
    def __init__(self, columns: dict, weights: list[float]):
        self.columns = columns
        self.weights = weights

    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        w = np.array(self.weights, dtype=float)
        w = w / w.sum()
        idx = gen.choice(len(self.weights), size=n, p=w)
        return {col: np.array(vals)[idx] for col, vals in self.columns.items()}

""" Sequential Distributions """

@dataclass
class Sequential(Distribution):
    def __init__(self, start: any, step: int = 1):
        self.start = start
        self.step: int = 1
    
    def _is_date(self):
        return isinstance(self.start, str)
 
    def sample_for_group(self, group_idx: int, n: int):
        if self._is_date():
            base = date.fromisoformat(self.start)
            val = str(base + timedelta(days=group_idx * self.step))
            return np.full(n, val, dtype=object)
        else:
            val = self.start + group_idx * self.step
            return np.full(n, val)
 
    def sample(self, n: int, gen: np.random.Generator) -> np.ndarray:
        if self._is_date():
            base = date.fromisoformat(self.start)
            return np.array([str(base + timedelta(days=i * self.step)) for i in range(n)], dtype=object)
        else:
            return np.arange(self.start, self.start + n * self.step, self.step)
 

""" Distribution Genarator Metho """
 
_CONTINUOUS = {"uniform", "normal", "lognormal", "weibull", "exponential", "beta", "gamma"}
 
def distribution_factory(d: dict) -> Distribution:
    t = d["type"]
    p = {k: v for k, v in d.items() if k != "type"}
    if t == "uniform":
        return Uniform(**p)
    elif t == "normal":
        return Normal(**p)
    elif t == "lognormal":
        return LogNormal(**p)
    elif t == "weibull":
        return Weibull(**p)
    elif t == "exponential":
        return Exponential(**p)
    elif t == "beta":
        return Beta(**p)
    elif t == "gamma":
        return Gamma(**p)
    elif t == "mixture":
        components = [distribution_factory(c) for c in p["components"]]
        return Mixture(components=components, weights=p["weights"])
    elif t == "weighted_choice":
        return WeightedChoice(values=p["values"], weights=p["weights"])
    elif t == "weighted_choice_mapping":
        return WeightedChoiceMapping(columns=p["columns"], weights=p["weights"])
    elif t == "sequential":
        return Sequential(**p)
    else:
        raise ValueError(f"Unknown distribution type: {t!r}")