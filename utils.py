import cv2


def draw_percent_steps(keypoints, gray, percent_on_step=10, delay=1000):
    step_size = int(len(keypoints) * (percent_on_step / 100))
    draw_step_size(keypoints, gray, step_size, delay)


def draw_fixed_steps(keypoints, gray, num_steps=20, delay=1000):
    step = len(keypoints) // num_steps
    draw_step_size(keypoints, gray, step, delay)


def draw_step_size(keypoints, gray, step=100, delay=1000):
    i = len(keypoints) // step
    if i > 0:
        while i > 0:
            i -= 1
            frame = cv2.drawKeypoints(gray, keypoints[:i * step], None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                      color=(150, 0, 0)
                                      )

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        image = cv2.drawKeypoints(gray, keypoints, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                  color=(150, 0, 0)
                                  )
        cv2.imshow('image', image)
        if cv2.waitKey() & 0xFF == ord('q'):
            cv2.destroyAllWindows()
