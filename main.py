import numpy as np
import cv2
from numba import njit, prange
import time


@njit(fastmath=True)
def intersect_cube(rox, roy, roz, rdx, rdy, rdz, size):
    eps = 1e-7

    # Считаем плоскости (просто числа)
    tx1 = (-size - rox) / (rdx + eps)
    tx2 = (size - rox) / (rdx + eps)
    ty1 = (-size - roy) / (rdy + eps)
    ty2 = (size - roy) / (rdy + eps)
    tz1 = (-size - roz) / (rdz + eps)
    tz2 = (size - roz) / (rdz + eps)

    # Ручной min/max без использования встроенных функций
    tmin_x = tx1 if tx1 < tx2 else tx2
    tmax_x = tx1 if tx1 > tx2 else tx2
    tmin_y = ty1 if ty1 < ty2 else ty2
    tmax_y = ty1 if ty1 > ty2 else ty2
    tmin_z = tz1 if tz1 < tz2 else tz2
    tmax_z = tz1 if tz1 > tz2 else tz2

    # Находим интервал пересечения
    near = tmin_x
    if tmin_y > near:
        near = tmin_y
    if tmin_z > near:
        near = tmin_z

    far = tmax_x
    if tmax_y < far:
        far = tmax_y
    if tmax_z < far:
        far = tmax_z

    if near > far or far < 0:
        return -1.0, 0.0, 0.0, 0.0

    # Определяем нормаль
    hx, hy, hz = rox + rdx * near, roy + rdy * near, roz + rdz * near
    nx, ny, nz = 0.0, 0.0, 0.0

    # Используем простую проверку вместо abs()
    d = 1e-3
    if hx > size - d:
        nx = 1.0
    elif hx < -size + d:
        nx = -1.0
    elif hy > size - d:
        ny = 1.0
    elif hy < -size + d:
        ny = -1.0
    elif hz > size - d:
        nz = 1.0
    elif hz < -size + d:
        nz = -1.0

    return near, nx, ny, nz


@njit(parallel=True, fastmath=True)
def render_frame(RES, t, cx, cy, cz, fx, fy, fz, rx, ry, rz, ux, uy, uz, x, y):
    img = np.zeros((RES, RES, 3), dtype=np.float32)

    # Вращение (разные скорости для объема)
    ay, ax = t * 1.3, t * 0.8
    sy, cy_m = np.sin(ay), np.cos(ay)
    sx, cx_m = np.sin(ax), np.cos(ax)

    # Три источника света (статичные, мировые)
    # 1. Основной (ярко-белый)
    l1x, l1y, l1z = 0.5, 0.8, 0.4
    # 2. Боковой (сероватый)
    l2x, l2y, l2z = -0.8, 0.1, 0.1
    # 3. Контровой (слабый)
    l3x, l3y, l3z = 0.1, -0.9, -0.2

    for i in prange(RES):
        for j in range(RES):
            # Вектор луча
            rdx_w = fx + x[0, j] * rx + y[i, 0] * ux
            rdy_w = fy + x[0, j] * ry + y[i, 0] * uy
            rdz_w = fz + x[0, j] * rz + y[i, 0] * uz
            rn = (rdx_w**2 + rdy_w**2 + rdz_w**2) ** 0.5
            rdx, rdy, rdz = rdx_w / rn, rdy_w / rn, rdz_w / rn

            # Трансформация в локальные координаты (Y затем X)
            # Вращаем луч
            tx, tz = rdx * cy_m + rdz * sy, -rdx * sy + rdz * cy_m
            rdx_l, rdy_l, rdz_l = tx, rdy * cx_m + tz * sx, -rdy * sx + tz * cx_m
            # Вращаем камеру
            tcx, tcz = cx * cy_m + cz * sy, -cx * sy + cz * cy_m
            rox_l, roy_l, roz_l = tcx, cy * cx_m + tcz * sx, -cy * sx + tcz * cx_m

            dist, nx, ny, nz = intersect_cube(
                rox_l, roy_l, roz_l, rdx_l, rdy_l, rdz_l, 1.0
            )

            if dist > 0:
                # Вращаем нормаль обратно в мир
                nx_t, nz_t = nx, ny * sx + nz * cx_m  # Сначала X
                ny_t = ny * cx_m - nz * sx

                nx_w = nx_t * cy_m - nz_t * sy  # Затем Y
                ny_w = ny_t
                nz_w = nx_t * sy + nz_t * cy_m

                # Освещение
                bright = 0.12  # Базовая освещенность

                # Добавляем свет от источников
                dot1 = nx_w * l1x + ny_w * l1y + nz_w * l1z
                if dot1 > 0:
                    bright += dot1 * 0.7

                dot2 = nx_w * l2x + ny_w * l2y + nz_w * l2z
                if dot2 > 0:
                    bright += dot2 * 0.3

                dot3 = nx_w * l3x + ny_w * l3y + nz_w * l3z
                if dot3 > 0:
                    bright += dot3 * 0.15

                # Обводка ребер (темная кайма)
                hx, hy, hz = (
                    rox_l + rdx_l * dist,
                    roy_l + rdy_l * dist,
                    roz_l + rdz_l * dist,
                )
                # Берем макс. координату (абсолютную)
                axc = hx if hx > 0 else -hx
                ayc = hy if hy > 0 else -hy
                azc = hz if hz > 0 else -hz
                edge = axc
                if ayc > edge:
                    edge = ayc
                if azc > edge:
                    edge = azc

                if edge > 0.97:
                    bright *= 0.5

                if bright > 1.0:
                    bright = 1.0
                img[i, j, :] = bright
            else:
                img[i, j, :] = 0.05
    return img


def main():
    RES = 800
    y, x = np.ogrid[1.0 : -1.0 : complex(RES), -1.0 : 1.0 : complex(RES)]
    t, last_t = 0.0, time.time()

    while True:
        dt = time.time() - last_t
        last_t, t = time.time(), t + dt

        # Параметры камеры
        cx, cy, cz = 0.0, 0.0, 5.0
        f = np.array([0.0, 0.0, -1.0])
        r = np.array([1.0, 0.0, 0.0])
        u = np.array([0.0, 1.0, 0.0])

        # Распаковываем массивы в отдельные float (для безопасности)
        frame = render_frame(
            RES,
            t,
            cx,
            cy,
            cz,
            f[0],
            f[1],
            f[2],
            r[0],
            r[1],
            r[2],
            u[0],
            u[1],
            u[2],
            x,
            y,
        )

        cv2.imshow("Volume White Cube", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
