"""Restricted zones drawn over the camera image (table `zones`)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    zone_id: int
    name: str
    polygon: tuple[tuple[float, float], ...]   # points from 0 to 1 relative to the image

    @classmethod
    def from_api(cls, data: dict) -> "Zone":
        points = tuple((float(x), float(y)) for x, y in data["coordinates"])
        return cls(int(data["zone_id"]), str(data["name"]), points)

    def contains(self, x: float, y: float) -> bool:
        """Ray casting: is the normalized point (x, y) inside the polygon?"""
        inside = False
        points = self.polygon
        j = len(points) - 1
        for i in range(len(points)):
            xi, yi = points[i]
            xj, yj = points[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside
