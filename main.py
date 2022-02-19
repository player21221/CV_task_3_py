# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/


import numpy as np
import cv2


def kernel_cv_to_conv(cvkernel, imsz):
    '''
    OpenCV kernel filter representation to usual convolution representation
    '''
    ky, kx = cvkernel.shape
    ny, nx = imsz
    h = np.zeros(imsz, dtype=np.float32)
    h1 = cvkernel[::-1, ::-1]
    h[0:ky, 0:kx] = h1
    h = np.roll(h, -(ky - 1) // 2, axis=1)
    h = np.roll(h, -(kx - 1) // 2, axis=0)
    return h


def deconvolve(cvkernel, blurred):
    ny,nx = blurred.shape
    h = kernel_cv_to_conv(cvkernel, blurred.shape)
    H = np.fft.fft2(h)
    Y = np.fft.fft2(blurred)
    Hsq = H**2
    noise = 1e-3
    U = Y * H.conj() / (np.abs(H) ** 2 + noise)
    u = np.fft.ifft2(U).real
    u = np.clip(u, 0, 255).astype(np.uint8)
    return u


#kern = np.array([[.125, .25, .125],
#                     [.25,  .5 , .25 ],
#                     [.125, .25, .125]])*.5
kern_1d = cv2.getGaussianKernel(9,2)
kern= np.transpose(kern_1d)*kern_1d
#kern=(1/np.sum(kerf))*kerf
image = cv2.imread("msg271576874-60545.jpg")

# for a,b in zip(range(0,10), range(0,10)):
#     pass

blue, green, red = np.dsplit(image, 3)
#dstb,dstg,dstr,resb,resg,resr=np.empty([6,image.shape[0],image.shape[1]])

dst=np.empty(blue.shape)
rest=np.empty(blue.shape)

for layer in [blue, green, red]:
    dstc = cv2.filter2D(layer,-1,kern)
    restc = deconvolve(kern,dstc)
    dst=np.append(dst,np.expand_dims(dstc, axis=2),axis=2)
    rest=np.append(rest,np.expand_dims(restc, axis=2),axis=2)

dst=np.uint8(np.delete(dst,0,axis=2))
rest=np.uint8(np.delete(rest,0,axis=2))

cv2.imwrite("blur.jpg", dst)
cv2.imwrite("restored.jpg", rest)


cv2.imshow("orig", image)
cv2.imshow("blur", dst)
cv2.imshow("rest", rest)
cv2.waitKey()


