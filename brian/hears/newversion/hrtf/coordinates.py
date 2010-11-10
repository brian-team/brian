from brian import *

__all__ = ['Coordinates',
           'CartesianCoordinates',
           'SphericalCoordinates',
           'AzimElev', 'AzimElevDegrees', 'AzimElevDistDegrees',
           ]

class Coordinates(ndarray):
    '''
    TODO: documentation
    '''
    names = None

    def convert_to(self, target):
        raise NotImplementedError
    @staticmethod
    def convert_from(source):
        raise NotImplementedError
    @property
    def system(self):
        return self.__class__
    @classmethod
    def make(cls, shape): #initialize a ndarray
        x = cls(shape=shape, dtype=cls.construct_dtype())
        return x
    @classmethod
    def construct_dtype(cls):
        return [(name, float) for name in cls.names]


class CartesianCoordinates(Coordinates):
    '''
    TODO: documentation
    '''
    names = ('x', 'y', 'z')

    def convert_to(self, target):
        if target is self.system:
            return self
        else:
            return target.convert_from(self)


class SphericalCoordinates(Coordinates):
    '''
    TODO: documentation
    '''
    names = ('r', 'theta', 'phi')


class AzimElev(Coordinates):
    '''
    TODO: documentation
    '''
    names = ('azim', 'elev')

    def convert_to(self, target):
        out = target.make(self.shape)
        if target is self.system:
            return self
        elif target is CartesianCoordinates:
            # Individual looking along x axis, ears at +- 1 on y axis, z vertical
            out['x'] = sin(self['azim']) * cos(self['elev'])
            out['y'] = cos(self['azim']) * cos(self['elev'])
            out['z'] = sin(self['elev'])
            return out
        elif target is AzimElevDegrees:
            azim = self['azim'] * 180 / pi
            azim[azim < 0] += 360
            out['azim'] = azim
            out['elev'] = self['elev'] * 180 / pi
            return out
        else:
            # Try to convert by going via Cartesian coordinates
            inter = self.convert_to(CartesianCoordinates)
            return target.convert_from(inter)
    @staticmethod
    def convert_from(source):
        if isinstance(source, AzimElev):
            return source
        elif isinstance(source, CartesianCoordinates):
            out = AzimElev.make(source.shape)
            x, y, z = source['x'], source['y'], source['z']
            r = sqrt(x ** 2 + y ** 2 + z ** 2)
            x /= r
            y /= r
            z /= r
            elev = arcsin(z / r)
            azim = arctan2(x, y)
            out['azim'] = azim
            out['elev'] = elev
            return out


class AzimElevDegrees(Coordinates):
    '''
    TODO: documentation
    '''
    names = ('azim', 'elev')

    def convert_to(self, target):
        if target is self.system:
            return self
        elif target is AzimElev:
            out = target.make(self.shape)
            out['azim'] = self['azim'] * pi / 180
            out['elev'] = self['elev'] * pi / 180
            return out
        else:
            inter = self.convert_to(AzimElev)
            return inter.convert_to(target)
    @staticmethod
    def convert_from(source):
        if isinstance(source, AzimElevDegrees):
            return source
        elif isinstance(source, CartesianCoordinates):
            inter = AzimElev.convert_from(source)
            return inter.convert_to(AzimElevDegrees)


class AzimElevDistDegrees(Coordinates):
    '''
    TODO: documentation
    '''
    names = ('azim', 'elev', 'dist')

    def convert_to(self, target):
        if target is self.system:
            return self
        elif target is CartesianCoordinates:
            out = target.make(self.shape)
            # Individual looking along x axis, ears at +- 1 on y axis, z vertical
            out['x'] = self['dist'] * sin(self['azim'] * pi / 180) * cos(self['elev'] * pi / 180)
            out['y'] = self['dist'] * cos(self['azim'] * pi / 180) * cos(self['elev'] * pi / 180)
            out['z'] = self['dist'] * sin(self['elev'] * pi / 180)
            return out
        elif target is AzimElev:
            out = target.make(self.shape)
            out['azim'] = self['azim'] * pi / 180
            out['elev'] = self['elev'] * pi / 180
            return out
        else:
            # Try to convert by going via Cartesian coordinates
            inter = self.convert_to(CartesianCoordinates)
            return target.convert_from(inter)
    @staticmethod
    def convert_from(source):
        if isinstance(source, AzimElevDegrees):
            return source
        elif isinstance(source, CartesianCoordinates):
            inter = AzimElev.convert_from(source)
            return inter.convert_to(AzimElevDegrees)
