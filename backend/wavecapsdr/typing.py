from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

NDArrayAny: TypeAlias = npt.NDArray[np.generic]
NDArrayFloat: TypeAlias = npt.NDArray[np.floating[Any]]
NDArrayComplex: TypeAlias = npt.NDArray[np.complexfloating[Any, Any]]
NDArrayInt: TypeAlias = npt.NDArray[np.integer[Any]]
