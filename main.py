import cv2
import numpy as np
import pynput.keyboard as kbd
import pynput.mouse as ms

import bodycr as cr
from bodycr.source.Modules.Drawer import Color

cap = cv2.VideoCapture(0)
WIDTH_SCREEN, HEIGHT_SCREEN = 1920, 1080

capture = cr.Recognize(all=cr.Prefabs.ALL.lite.Mount())
draw = cr.Drawer()
fps = cr.FPS()
keyboard = kbd.Controller()
mouse = ms.Controller()

WIDTH, HEIGHT = cr.Resolutions.VGA
LEFT_MARGIN = int(WIDTH / 2) - 100
RIGHT_MARGIN = int(WIDTH / 2) + 100
DOWN_MARGIN = HEIGHT - 1080

smooth = 5

while True:
    succes, img = cap.read()
    img = cv2.resize(img, cr.Resolutions.VGA)
    img = cv2.flip(img, 1)

    capture.Read(img, cr.DETECT_ALL)
    draw.UpdateImage(img)
    draw.DrawComponent(capture.pose, cr.POSE_CONNECTIONS)
    for hand in capture.hands:
        draw.DrawComponent(hand.landmarks, cr.HAND_CONNECTIONS)

    # ESQUERDA
    draw.PutLine(
        cr.Mathb.TupToPoint((LEFT_MARGIN, 0)), cr.Point(LEFT_MARGIN, HEIGHT), Color.blue
    )
    draw.PutText("LEFT", cr.Mathb.TupToPoint((50, 70)), 2, Color.red, 2)

    # DIREITA
    draw.PutLine(
        cr.Mathb.TupToPoint((RIGHT_MARGIN, 0)),
        cr.Point(RIGHT_MARGIN, HEIGHT),
        Color.blue,
    )
    draw.PutText("RIGHT", cr.Mathb.TupToPoint((430, 70)), 2, Color.red, 2)

    # JUMP
    draw.PutLine(
        cr.Mathb.TupToPoint((0, DOWN_MARGIN)), cr.Point(WIDTH, DOWN_MARGIN), Color.blue
    )
    draw.PutText(
        "JUMP", cr.Mathb.TupToPoint((int(WIDTH / 2) - 75, 430)), 2, Color.red, 2
    )

    if len(capture.pose) != 0:
        if capture.pose[31].y < DOWN_MARGIN and capture.pose[32].y < DOWN_MARGIN:
            draw.PutText(
                "JUMP",
                cr.Mathb.TupToPoint((int(WIDTH / 2) - 75, 430)),
                2,
                Color.green,
                2,
            )
            keyboard.press(kbd.Key.space)
        else:
            keyboard.release(kbd.Key.space)

        if capture.pose[0].x < LEFT_MARGIN:
            draw.PutText("LEFT", cr.Mathb.TupToPoint((50, 70)), 2, Color.green, 2)
            keyboard.press("a")
        else:
            keyboard.release("a")

        if capture.pose[0].x > RIGHT_MARGIN:
            draw.PutText("RIGHT", cr.Mathb.TupToPoint((430, 70)), 2, Color.green, 2)
            keyboard.press("d")
        else:
            keyboard.release("d")
    else:
        pass

    if len(capture.leftHand.landmarks) != 0:
        armAngle = cr.Mathb.GetAngle(
            capture.leftHand.position, capture.pose[12], capture.pose[24]
        )
        R = 50

        if capture.leftHand.position.x < capture.pose[12].x:
            armAngle = 360 - armAngle

        armAngle = np.radians(armAngle)

        armX = R * np.sin(armAngle) + int(WIDTH / 2)
        armY = R * np.cos(armAngle) + int(HEIGHT / 2)

        draw.PutCircle(cr.Mathb.TupToPoint((armX, armY)), 25, draw.FILL, Color.magenta)

        armXinterp = np.interp(armX, (0, WIDTH), (0, WIDTH_SCREEN))
        armYinterp = np.interp(armY, (0, HEIGHT), (0, HEIGHT_SCREEN))

        mouse.position = (int(armXinterp), int(armYinterp))

        closed = capture.leftHand.GetClosedFingers()

        if closed[4] and closed[3] and closed[2] and not closed[1]:
            mouse.click(ms.Button.left)
            draw.PutCircle(
                cr.Mathb.TupToPoint((armX, armY)), 25, draw.FILL, Color.green
            )

    fps.Update(img)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
