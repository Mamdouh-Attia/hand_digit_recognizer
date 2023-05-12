from skimage import feature
from skimage.feature import local_binary_pattern

class Features:
    points=24
    radius=3
    orientations=8
    scales=5 
    sigma=1.0
    gamma=0.5
    def __init__(self):
        pass
    def extract_hog_features(self,image):
        hog_features, _ = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, visualize=True, multichannel=False)
        return hog_features

    def extract_lbp_features(self,image):
        lbp = local_binary_pattern(image, self.points, self.radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.points + 3), range=(0, self.points + 2))
        hist = hist.astype('float')
        hist /= (hist.sum() + 1e-7)
        return hist

    def gabor_features(self,image):
        gabor_features = []
        for orientation in np.linspace(0, np.pi, self.orientations, endpoint=False):
            for scale in range(self.scales):
                freq = 1 / (0.5 * (scale + 1))
                gabor_filter = cv2.getGaborKernel((21, 21), self.sigma, orientation, freq, self.gamma, 0, ktype=cv2.CV_32F)
                filtered_image = ndimage.convolve(image, gabor_filter)
                gabor_features.append(filtered_image.mean())
                gabor_features.append(filtered_image.var())
        return np.array(gabor_features)
