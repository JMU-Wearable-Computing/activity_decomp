import math


class Distance():
    
    def __call__(self, actual: dict, expected: dict) -> dict:
        pass

class EuclideanDistance(Distance):

    def __call__(self, actual: dict, expected: dict) -> dict:
        out = {}
        squared_total = 0
        for key in actual.keys():
            if key not in actual or key not in expected:
                out[key] = None
            else:
                out[key] = actual[key] - expected[key]
                squared_total += out[key] ** 2

        out["all"] = math.sqrt(squared_total)

        return out
        