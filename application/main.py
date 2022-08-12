from multiprocessing.dummy import Manager
from numpy import angle
from kivy.core.text import LabelBase
from kivy.uix.screenmanager import ScreenManager, NoTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivymd.uix.list import OneLineAvatarIconListItem
from kivymd.uix.button import MDFlatButton
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
Window.size = (350, 600)
from kivymd.uix.dialog import MDDialog
from kivy.clock import Clock
from kivy.animation import Animation
import random
import pickle
from model.models import *


class ItemConfirm(OneLineAvatarIconListItem):
    divider = None
    def set_icon(self, instance_check):
        instance_check.active = True
        check_list = instance_check.get_widgets(instance_check.group)
        for check in check_list:
            if check != instance_check:
                check.active = False


class ContentNavigationDrawer(BoxLayout):
    manager = ObjectProperty()
    nav_drawer = ObjectProperty()       


class export_app(MDApp):
    input_symbol = ""
    i = 0
    angle = 45
    stop_1 = random.randrange(20, 50)
    stop_2 = random.randrange(65, 90)
     
    def build(self):
        global screen_manager
        screen_manager = ScreenManager(transition=NoTransition())
        screen_manager.add_widget(Builder.load_file("start.kv"))
        screen_manager.add_widget(Builder.load_file("loading.kv"))
        Clock.schedule_interval(self.loader, 0.05)
        screen_manager.add_widget(Builder.load_file("final.kv"))
        return screen_manager
    
    def search_result(self):
        df = pd.read_csv('./data/bi_data.csv')
        data = df[['KOR_export']]
        self.AI = AI_Korea_Export(data) 
        self.i = 0
        screen_manager.get_screen("loading").ids.progress_bar.value = 0
        screen_manager.get_screen("loading").ids.progress_bar.color = (0, 0, 0, 0)
        Clock.schedule_interval(self.loader, 0.05)
        screen_manager.current = "loading"
               
        
    def loading(self, *args):
        anim = Animation(height=30, width=30, spacing=[5, 5], duration=0.02)
        anim += Animation(height=30, width=30, spacing=[5, 5], duration=0.02)
        anim += Animation(angle = self.angle, duration=0.02)
        anim.bind(on_complete=self.loading)
        anim.start(screen_manager.get_screen("loading").ids.loading)
        self.angle += 45

    def close_dialog(self, obj):
        self.dialog.dismiss()
            
    def close_dialog_1(self, obj):
        self.dialog_1.dismiss()
        
    def on_start(self):
        self.loading()
    
    def loader(self, *args):
        try:
            self.i += 1
            screen_manager.get_screen("loading").ids.progress_bar.value = self.i
            screen_manager.get_screen("loading").ids.progress_bar.color = (1, 1, 1, 1)
            if self.i == self.stop_1:
                self.short_result = self.AI.short_term_predict() # {symbol = [train_acc, test_acc, prediction]}
            elif self.i == self.stop_2:
                self.long_result = self.AI.long_term_predict() # {symbol : [short-MA, long-MA, current position]}
        except:
            Clock.unschedule(self.loader)
            
    def short_forcast(self):
        item_list = [ItemConfirm(text="Volume {}".format(self.short_result['KOR_export'][0]))]
        self.dialog = MDDialog(title='Next Month Korea Export',
                               type="confirmation",
                               items = item_list,
                               width_offset = "8dp",
                               buttons=[MDFlatButton(text='CLOSE',
                                                     theme_text_color="Custom",
                                                     text_color=self.theme_cls.primary_color,
                                                     on_release=self.close_dialog)]
                                   )
        self.dialog.open()

    def long_forcast(self):
        item_list = [ItemConfirm(text="{} !".format(self.long_result['KOR_export'][2]))]
        self.dialog_1 = MDDialog(title='Long-Term Export Signal',
                               type="simple",
                               width_offset = "5dp",
                               items = item_list,
                               buttons=[MDFlatButton(text='CLOSE',
                                                       theme_text_color="Custom",
                                                       text_color=self.theme_cls.primary_color,
                                                       on_release=self.close_dialog_1)]
                                   )
        self.dialog_1.open()

    
    
if __name__ == "__main__":
    LabelBase.register(name="Anton", fn_regular="AlumniSansCollegiateOne-Regular.ttf")
    LabelBase.register(name="Ubuntu", fn_regular="Cairo-Regular.ttf")
    
    export_app().run()