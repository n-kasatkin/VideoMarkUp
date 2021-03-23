class TrackNumberSegments:
    def __init__(self, segments=None):
        # Segments should be always sorted
        self.segments = segments or []

    def add_segment(self, new_first, new_last, new_number):
        to_remove, new_segments = [], []
        for si, (first, last, number) in enumerate(self.segments):
            # Nothing to add more
            if new_first > new_last:
                break

            # No intersection
            if last < new_first or new_last < first:
                continue

            if first < new_first:
                new_segments.append([first, new_first - 1, number])
                first = new_first

            # Case 1
            # New ...]
            # Old .....]
            if new_last <= last:
                new_segments.append([new_first, new_last, new_number])
                if new_last < last:
                    new_segments.append([new_last + 1, last, number])

            # Case 2
            # New .....]
            # Old ...]
            else:
                new_segments.append([new_first, last, new_number])

            new_first = last + 1
            to_remove.append(si)

        # Add rest of new segment
        if new_first <= new_last:
            new_segments.append([new_first, new_last, new_number])

        # Merge new segments
        new_segments = self._merge(new_segments)

        # Remove
        segments = [seg for si, seg in enumerate(self.segments) if si not in to_remove]

        # Update segments
        self.segments = sorted(segments + new_segments)

    def get_number(self, frame_no):
        # No numbers
        if len(self.segments) == 0:
            return None

        # frame_no outside segments
        if frame_no < self.segments[0][0] or frame_no > self.segments[-1][1]:
            return None

        # Bin search
        l, r = 0, len(self.segments)
        while l < r:
            m = l + (r - l) // 2
            first, last, number = self.segments[m]
            if first <= frame_no <= last:
                return number
            elif frame_no < first:
                r = m
            else:
                l = m + 1

        return None

    def _merge(self, segments):
        if len(segments) == 0:
            return segments

        result = []
        new_first, new_last, new_number = segments[0]
        for i in range(1, len(segments)):
            first, last, number = segments[i]
            if number != new_number:
                result.append([new_first, new_last, new_number])
                new_first, new_last, new_number = first, last, number
            else:
                new_last = last

        result.append([new_first, new_last, new_number])

        return result


def test_add_segment():
    segments = TrackNumberSegments()
    assert segments.segments == []

    segments.add_segment(0, 5, 2)
    assert segments.segments == [[0, 5, 2]]

    segments.add_segment(6, 10, -1)
    assert segments.segments == [[0, 5, 2], [6, 10, -1]]

    segments.add_segment(4, 7, 2)
    assert segments.segments == [[0, 7, 2], [8, 10, -1]]

    segments.add_segment(15, 15, 1)
    assert segments.segments == [[0, 7, 2], [8, 10, -1], [15, 15, 1]]

    segments.add_segment(5, 20, 3)
    assert segments.segments == [[0, 4, 2], [5, 20, 3]]

    segments.add_segment(13, 13, -1)
    assert segments.segments == [[0, 4, 2], [5, 12, 3], [13, 13, -1], [14, 20, 3]]

    segments.add_segment(-5, 1, -1)
    assert segments.segments == [[-5, 1, -1], [2, 4, 2], [5, 12, 3], [13, 13, -1], [14, 20, 3]]

    segments.add_segment(7, 18, 4)
    assert segments.segments == [[-5, 1, -1], [2, 4, 2], [5, 6, 3], [7, 18, 4], [19, 20, 3]]


def test_get_number():
    segments = TrackNumberSegments()

    segments.add_segment(0, 5, 3)
    for i in range(-3, 0):
        assert segments.get_number(i) is None
    for i in range(6):
        assert segments.get_number(i) == 3
    for i in range(6, 9):
        assert segments.get_number(i) is None

    segments.add_segment(9, 14, -1)
    for i in range(-3, 0):
        assert segments.get_number(i) is None
    for i in range(6):
        assert segments.get_number(i) == 3
    for i in range(6, 9):
        assert segments.get_number(i) is None
    for i in range(9, 15):
        assert segments.get_number(i) == -1
    for i in range(15, 20):
        assert segments.get_number(i) is None

    segments.add_segment(7, 7, 2)
    for i in range(-3, 0):
        assert segments.get_number(i) is None
    for i in range(6):
        assert segments.get_number(i) == 3
    for i in range(6, 7):
        assert segments.get_number(i) is None
    for i in range(7, 8):
        assert segments.get_number(i) == 2
    for i in range(8, 9):
        assert segments.get_number(i) is None
    for i in range(9, 15):
        assert segments.get_number(i) == -1
    for i in range(16, 20):
        assert segments.get_number(i) is None


def test():
    test_add_segment()
    test_get_number()


if __name__ == '__main__':
    test()
