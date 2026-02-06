from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Polygon
from matplotlib.path import Path


@dataclass
class AffineTransform:
    """Represents T(x) = Ax + b"""

    linear: npt.NDArray[np.float64]  # Shape (d, d)
    bias: npt.NDArray[np.float64]  # Shape (d,)


class FractalDatasetGenerator:
    """
    A class to generate fractal datasets using Iterated Function Systems (IFS).
    Based on Malach and Shalev-Shwartz (2019).
    """

    def __init__(
        self,
        dimensionality: int,
        transforms: list[AffineTransform] | None = None,
        base_shape: npt.NDArray[np.float64] | None = None,
        seed: int | None = None,
    ):
        """Initializes the fractal generator with either a provided transformation tensor or generates a random one.

        Args:
            dimensionality (int): The dimension of the space.
            transforms (list[AffineTransform] | None, optional): A list of AffineTransform objects defining the IFS. Defaults to None.
            base_shape (npt.NDArray[np.float64] | None, optional): Optional base shape defined by its vertices. If None, defaults to the hypercube [-1, 1]^d. Defaults to None.
            seed (int | None, optional): Random seed for reproducibility. Defaults to None.
        """
        self.d = dimensionality
        self.rng = np.random.default_rng(seed)

        if base_shape is not None:
            self.base_shape = base_shape
        else:
            # Default hypercube [-1, 1]^d
            self.base_shape = np.array(
                [[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64
            )
        self.base_path = Path(self.base_shape)
        self.bounds_min = self.base_shape.min(axis=0)
        self.bounds_max = self.base_shape.max(axis=0)

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._generate_random_fractal() @ staticmethod

    def from_tensor(tensor: npt.NDArray[np.float64]) -> list[AffineTransform]:
        """Helper to convert a (N, d, d+1) tensor into a list of AffineTransforms."""
        assert tensor.ndim == 3, "Tensor must be 3D (num_transforms, d, d+1)"
        assert (
            tensor.shape[2] == tensor.shape[1] + 1
        ), "Last dimension must be d+1 (linear + bias)"
        transforms = []
        for i in range(tensor.shape[0]):
            transforms.append(
                AffineTransform(linear=tensor[i, :, :-1], bias=tensor[i, :, -1])
            )
        return transforms

    def _generate_random_fractal(self) -> list[AffineTransform]:
        """
        Generates random contractive affine transformations.
        Uses a grid heuristic to ensure images of K0 do not overlap.
        """
        # Divide space into a grid (e.g., 3^d cells) to guarantee non-overlap
        grid_divisions = 3
        cell_width = (self.bounds_max - self.bounds_min) / grid_divisions

        # Generate all possible grid positions
        grid_indices = np.stack(
            np.meshgrid(*[np.arange(grid_divisions) for _ in range(self.d)]), -1
        ).reshape(-1, self.d)

        # Randomly select a subset of cells (between 2 and grid_count-1)
        num_transforms = self.rng.integers(2, max(3, len(grid_indices) // 2))
        chosen_indices = self.rng.choice(
            len(grid_indices), size=num_transforms, replace=False
        )
        chosen_cells = grid_indices[chosen_indices]

        transforms = []
        scale_factor = 1.0 / grid_divisions

        # Create a transform mapping K0 [-1, 1] -> Grid Cell
        # T(x) = (scale * x) + offset
        for cell_idx in chosen_cells:
            # Calculate center of the target grid cell
            # Map grid index [0, 1, 2] to [-1, 1] coordinate space
            # Cell 0 center: -1 + width/2
            cell_center = self.bounds_min + (cell_idx * cell_width) + (cell_width / 2.0)

            # Linear part: simple uniform scaling (could add random rotation here)
            linear = np.eye(self.d) * scale_factor

            transforms.append(AffineTransform(linear, cell_center))

        return transforms

    def sample_positive(self, depth: int, num_examples: int) -> npt.NDArray[np.float64]:
        """Samples positive examples from K_n (fractal at specific depth).

        Args:
            depth (int): Fractal depth to sample from.
            num_examples (int): Number of positive examples to generate.

        Returns:
            npt.NDArray[np.float64]: Array of sampled positive examples with shape (num_examples, dimensionality).
        """
        final_points = np.zeros((0, self.d))
        batch_size = num_examples * 2

        while len(final_points) < num_examples:
            # Initialize random points in bounding box
            pts = self.rng.uniform(
                self.bounds_min, self.bounds_max, size=(batch_size, self.d)
            )

            # Filter to points inside base shape
            valid_seeds = self.base_path.contains_points(pts)
            pts = pts[valid_seeds]
            current_batch_size = len(pts)

            # Apply IFS
            cum_linear = np.tile(np.eye(self.d), (current_batch_size, 1, 1))
            cum_bias = np.zeros((current_batch_size, self.d))

            # Randomly apply transforms for 'depth' iterations
            for _ in range(depth):
                idx = self.rng.integers(
                    0, len(self.transforms), size=current_batch_size
                )
                step_linears = np.stack([self.transforms[i].linear for i in idx])
                step_biases = np.stack([self.transforms[i].bias for i in idx])
                cum_bias = (step_linears @ cum_bias[:, :, np.newaxis]).squeeze(
                    2
                ) + step_biases
                cum_linear = step_linears @ cum_linear
            pts = (cum_linear @ pts[:, :, np.newaxis]).squeeze(2) + cum_bias

            final_points = np.vstack([final_points, pts])

        return final_points[:num_examples]

    def sample_negative_at_level(
        self, depth: int, num_examples: int
    ) -> npt.NDArray[np.float64]:
        """Samples negative examples from the gaps created at a specific depth.

        Args:
            depth (int): Fractal depth to sample from (negative examples are from the gap between K_{depth-1} and K_{depth}).
            num_examples (int): Number of negative examples to generate.

        Returns:
            npt.NDArray[np.float64]: Negative examples sampled from the gaps.
        """
        assert depth >= 1, "Negative examples (gaps) only exist for depth >= 1"

        final_points = np.zeros((0, self.d))
        batch_size = num_examples * 5  # High rejection rate likely in gaps

        while len(final_points) < num_examples:
            # Initialize random points in bounding box
            pts = self.rng.uniform(
                self.bounds_min, self.bounds_max, size=(batch_size, self.d)
            )

            # Must be inside base shape (K0)
            in_parent = self.base_path.contains_points(pts)

            # Must NOT be inside K1 (Union of T_i(K0))
            # x in T_i(K0) <==> T_i^{-1}(x) in K0
            in_any_child = np.zeros(batch_size, dtype=bool)
            for t in self.transforms:
                try:
                    inv_linear = np.linalg.inv(t.linear)
                except np.linalg.LinAlgError:
                    continue

                # Map seed back to K0 space for this transform
                # pre_image = A^{-1}(x - b)
                centered = pts - t.bias
                pre_images = centered @ inv_linear.T

                # Check if pre_image is in K0
                in_this_child = self.base_path.contains_points(pre_images)
                in_any_child = in_any_child | in_this_child

            # Valid Seeds: In Parent AND Not In Any Child
            valid_seeds_mask = in_parent & (~in_any_child)
            pts = pts[valid_seeds_mask]
            current_batch_size = len(pts)

            if current_batch_size == 0:
                continue

            # Randomly apply transforms for 'depth-1' iterations
            cum_linear = np.tile(np.eye(self.d), (current_batch_size, 1, 1))
            cum_bias = np.zeros((current_batch_size, self.d))
            for _ in range(depth - 1):
                idx = self.rng.integers(
                    0, len(self.transforms), size=current_batch_size
                )
                step_linears = np.stack([self.transforms[i].linear for i in idx])
                step_biases = np.stack([self.transforms[i].bias for i in idx])
                cum_bias = (step_linears @ cum_bias[:, :, np.newaxis]).squeeze(
                    2
                ) + step_biases
                cum_linear = step_linears @ cum_linear

            pts = (cum_linear @ pts[:, :, np.newaxis]).squeeze(2) + cum_bias
            final_points = np.vstack([final_points, pts])

        return final_points[:num_examples]

    def sample_negative(
        self, depth: int, num_examples: int, difficulty: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Samples negative examples from the gaps created at all levels up to a specific depth, with a difficulty parameter to control the distribution.

        Args:
            depth (int): Fractal depth to sample from (negative examples are from the gap between K_{d-1} and K_{d} for d in [1, depth]).
            num_examples (int): Number of negative examples to generate.
            difficulty (float): A value in [0, 1] controlling the distribution of negative samples across levels. 0 means all from the largest gap (depth=1), 1 means uniform across all levels.

        Returns:
            npt.NDArray[np.float64]: Negative examples sampled from the gaps across all levels.
        """
        assert depth >= 1, "Negative examples (gaps) only exist for depth >= 1"
        assert 0.0 <= difficulty <= 1.0, "Difficulty must be in [0, 1]"

        # Calculate number of points to take from each level based on difficulty
        if difficulty == 1:
            level_probs = np.ones(depth) / depth
        elif difficulty == 0:
            level_probs = np.zeros(depth)
            level_probs[0] = 1.0
        else:
            level_probs = difficulty ** np.arange(depth)
            level_probs /= level_probs.sum()
        num_at_level = (level_probs * num_examples).astype(int)

        # Sample negative points at each level
        all_negatives = []
        for d, n in zip(range(1, depth + 1), num_at_level):
            all_negatives.append(self.sample_negative_at_level(d, n))
        all_negatives = np.vstack(all_negatives)

        return all_negatives

    def render(self, depth: int, ax: plt.Axes) -> None:
        """Visualizes the fractal at the given depth. Only supported for dimensionality=2.

        Args:
            depth (int): Fractal depth.
            ax (plt.Axes): Matplotlib axes object on which to render the fractal.
        """

        assert self.d == 2, f"Rendering only supported for 2D (current dim={self.d})"

        # Determine render bounds
        b_min = self.base_shape.min(axis=0)
        b_max = self.base_shape.max(axis=0)
        span = b_max - b_min
        ax.set_xlim(b_min[0] - span[0] * 0.02, b_max[0] + span[0] * 0.02)
        ax.set_ylim(b_min[1] - span[1] * 0.02, b_max[1] + span[1] * 0.02)
        ax.set_aspect("equal")

        queue = [AffineTransform(np.eye(2), np.zeros(2))]
        for _ in range(depth):
            next_queue = []
            for parent_t in queue:
                for t in self.transforms:
                    new_linear = parent_t.linear @ t.linear
                    new_bias = parent_t.linear @ t.bias + parent_t.bias
                    next_queue.append(AffineTransform(new_linear, new_bias))
            queue = next_queue

        for t in queue:
            transformed_poly = self.base_shape @ t.linear.T + t.bias
            poly = Polygon(
                transformed_poly,
                closed=True,
                edgecolor="blue",
                facecolor="lightblue",
                alpha=0.2,
            )
            ax.add_patch(poly)

    @staticmethod
    def cantor_2d():
        # Four corner squares with Scale=1/3
        cantor_transforms = [
            # Top Left: Scale by 1/3, Shift to (-2/3, 2/3)
            AffineTransform(
                linear=np.array([[1 / 3, 0], [0, 1 / 3]]),
                bias=np.array([-2 / 3, 2 / 3]),
            ),
            # Top Right: Scale by 1/3, Shift to (2/3, 2/3)
            AffineTransform(
                linear=np.array([[1 / 3, 0], [0, 1 / 3]]), bias=np.array([2 / 3, 2 / 3])
            ),
            # Bottom Left: Scale by 1/3, Shift to (-2/3, -2/3)
            AffineTransform(
                linear=np.array([[1 / 3, 0], [0, 1 / 3]]),
                bias=np.array([-2 / 3, -2 / 3]),
            ),
            # Bottom Right: Scale by 1/3, Shift to (2/3, -2/3)
            AffineTransform(
                linear=np.array([[1 / 3, 0], [0, 1 / 3]]),
                bias=np.array([2 / 3, -2 / 3]),
            ),
        ]
        return FractalDatasetGenerator(dimensionality=2, transforms=cantor_transforms)

    @staticmethod
    def sierpinski_triangle():
        # Three corner triangles with Scale=1/2
        base_triangle = np.array([[0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]])
        sierp_transforms = []
        for vertex in base_triangle:
            sierp_transforms.append(
                AffineTransform(linear=np.eye(2) * 0.5, bias=vertex * 0.5)
            )
        return FractalDatasetGenerator(
            dimensionality=2, transforms=sierp_transforms, base_shape=base_triangle
        )

    @staticmethod
    def vicsek():
        # A cross shape with Scale=1/3
        vicsek_transforms = [
            # Center
            AffineTransform(linear=np.eye(2) * 1 / 3, bias=np.array([0.0, 0.0])),
            # Up
            AffineTransform(linear=np.eye(2) * 1 / 3, bias=np.array([0.0, 2 / 3])),
            # Down
            AffineTransform(linear=np.eye(2) * 1 / 3, bias=np.array([0.0, -2 / 3])),
            # Left
            AffineTransform(linear=np.eye(2) * 1 / 3, bias=np.array([-2 / 3, 0.0])),
            # Right
            AffineTransform(linear=np.eye(2) * 1 / 3, bias=np.array([2 / 3, 0.0])),
        ]
        return FractalDatasetGenerator(dimensionality=2, transforms=vicsek_transforms)

    @staticmethod
    def pentaflake():
        # A pentagon shape with Scale≈0.382
        r_penta = (3 - np.sqrt(5)) / 2
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]
        base_pentagon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        penta_transforms = []
        for vertex in base_pentagon:
            # Scale and move to corner
            penta_transforms.append(
                AffineTransform(linear=np.eye(2) * r_penta, bias=vertex * (1 - r_penta))
            )
        return FractalDatasetGenerator(
            dimensionality=2, transforms=penta_transforms, base_shape=base_pentagon
        )


if __name__ == "__main__":
    # A demo
    num_examples = 5000
    fractals = [
        ("Sierpinski", FractalDatasetGenerator.sierpinski_triangle(), 4),
        ("Vicsek", FractalDatasetGenerator.vicsek(), 3),
        ("Pentaflake", FractalDatasetGenerator.pentaflake(), 3),
        ("Cantor", FractalDatasetGenerator.cantor_2d(), 3),
    ]
    fig, axs = plt.subplots(1, len(fractals), figsize=(5 * len(fractals), 5))
    for i, (name, fractal, depth) in enumerate(fractals):
        pos = fractal.sample_positive(depth, num_examples)
        neg = fractal.sample_negative(depth, num_examples, difficulty=0.7)
        ax = axs[i]
        fractal.render(depth=depth, ax=ax)
        ax.scatter(pos[:, 0], pos[:, 1], alpha=0.3, s=1, label="positive")
        ax.scatter(neg[:, 0], neg[:, 1], alpha=0.3, s=1, label="negative")
        ax.set(xticks=[], yticks=[], title=f"{name} (depth={depth})")
    fig.tight_layout()
    plt.show()
