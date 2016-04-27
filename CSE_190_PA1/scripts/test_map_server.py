import unittest
from map_server import MapServer


class TestMapServer(unittest.TestCase):

    def test_0(self):
        ms = MapServer()
        self.assertEquals(ms.pipe_map[0][0], 'C')

if __name__ == '__main__':
    unittest.main()
