import pygame
import numpy as np
import writescript
pygame.init()

# Main loop
def take_input(is_writing):
    
    screen = pygame.display.set_mode((280, 280))
    clock = pygame.time.Clock()
    drawing = False
    teach=None
    
    while True:
        clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                # Get pixels
                small_surface = pygame.transform.smoothscale(screen, (28, 28))
                pixels = pygame.surfarray.array3d(small_surface)
                gray = (
                        0.299 * pixels[:, :, 0] +
                        0.587 * pixels[:, :, 1] +
                        0.114 * pixels[:, :, 2]
                        )
                gray= gray/255
                x = gray.reshape(28 * 28, 1)
                
                if event.key == pygame.K_2:
                    teach=True
                    print (int(teach))
                    
                if event.key == pygame.K_9:
                    teach=False
                    print (int(teach))
                if not is_writing: pygame.quit()
                
                if event.key == pygame.K_s:
                    return x
                elif event.key ==pygame.K_1:
                    return x, 1
                elif event.key ==pygame.K_0:
                    return x,0
                
                
                elif event.key==pygame.K_d:
                    originial=screen.copy()
                    
                    for i in range(100):
                        choice=np.random.randint(0,3)
                        tmp=screen.copy()
                        if choice==0:
                            aug=pygame.transform.rotate(originial,np.random.randint(-45,45))
                            aug_rect = aug.get_rect()
                            aug_rect.center = (140, 140)
                            screen.blit(aug,aug_rect.topleft)
                        elif choice==1:
                            screen.blit(tmp,(np.random.randint(-1,2),np.random.randint(-1,2)))
                            
                        elif choice==2:
                            aur=pygame.transform.rotozoom(originial,np.random.randint(-45,45),np.random.uniform(0.75,1.25))
                            aur_rect = aur.get_rect()
                            aur_rect.center = (140, 140)
                            screen.blit(aur,aur_rect.topleft)
                            
                        small_surface = pygame.transform.smoothscale(screen, (28, 28))
                        pixels = pygame.surfarray.array3d(small_surface)
                        gray = (
                                0.299 * pixels[:, :, 0] +
                                0.587 * pixels[:, :, 1] +
                                0.114 * pixels[:, :, 2]
                                )
                        gray= gray/255
                        m = gray.reshape(28 * 28, 1)
                        
                        writescript.writedata(m,int(teach))
                        clock.tick(100)
                        pygame.display.update()
                            
                    
        
        if drawing:
            pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (255, 255, 255), pos, 8)
        
        pygame.display.update()
        
